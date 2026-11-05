"""Quadratic/quartic attitude laws and deterministic projective integration.

The density, observations, and coordinate Jacobian are evaluated exactly.
Local Gauss--Hermite plus global Haar multiple-importance quadrature is an
integration rule, not a Gaussian attitude-state law.
Finite quadrature and finite forward-KL projection remain explicit numerical
approximations. No candidates are sampled and no covariance floor is applied.
"""
from functools import lru_cache
import itertools
import math

import numpy as np
from scipy.special import logsumexp

from probtf.bingham import _rotation_quadratic_forms


@lru_cache(maxsize=2)
def symmetric_tracefree_basis(size):
    result = []
    for k in range(1, size):
        diagonal = np.zeros(size)
        diagonal[:k] = 1. / math.sqrt(k * (k + 1))
        diagonal[k] = -k / math.sqrt(k * (k + 1))
        result.append(np.diag(diagonal))
    for i in range(size):
        for j in range(i + 1, size):
            value = np.zeros((size, size))
            value[i, j] = value[j, i] = 1. / math.sqrt(2.)
            result.append(value)
    return np.asarray(result)


B4 = symmetric_tracefree_basis(4)
B3 = symmetric_tracefree_basis(3)
ROTATION_FORMS = np.asarray(_rotation_quadratic_forms()).reshape(3, 3, 4, 4)


def _symmetric(value, shape):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError('Expected finite matrix with shape ' + str(shape))
    return .5 * (array + array.T)


def _quaternion(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError('A reference quaternion needs four finite values')
    norm = float(np.linalg.norm(value))
    if norm == 0.:
        raise ValueError('A reference quaternion must be nonzero')
    return value / norm


def projective_basis(reference):
    """Columns q_ref*(0,e_i), with Hamilton product and wxyz order."""
    w, x, y, z = _quaternion(reference)
    return np.array([[-x, -y, -z], [w, -z, y], [z, w, -x], [-y, x, w]])


def _projective_bases(references):
    w,x,y,z=np.asarray(references,float).T
    return np.stack((-x,-y,-z,w,-z,y,z,w,-x,-y,x,w),axis=-1).reshape(-1,4,3)


@lru_cache(maxsize=1)
def _spatial_mode_seeds():
    """Twenty-four fixed, spatially distributed projective search directions.

    These are numerical optimizer starts, never attitude-law samples. They
    combine coordinate axes, signed coordinate pairs and hypercube diagonals.
    Unlike selecting only the highest Haar-grid values, they retain starts in
    separated basins even when an extremely narrow peak has a low grid score.
    """
    values=list(np.eye(4))
    for i in range(4):
        for j in range(i+1,4):
            for sign in (-1.,1.):
                q=np.zeros(4);q[i]=1./math.sqrt(2.);q[j]=sign/math.sqrt(2.)
                values.append(q)
    values.extend(np.r_[1.,signs]/2. for signs in itertools.product((-1.,1.),repeat=3))
    result=np.asarray(values);result.flags.writeable=False
    return result


@lru_cache(maxsize=12)
def _hermite_grid(order):
    if isinstance(order, bool) or int(order) != order or order < 3:
        raise ValueError('Quadrature order must be an integer at least three')
    nodes, weights = np.polynomial.hermite.hermgauss(int(order))
    grids = np.meshgrid(nodes, nodes, nodes, indexing='ij')
    coordinates = np.stack([v.ravel() for v in grids], axis=1)
    logs = np.meshgrid(np.log(weights), np.log(weights), np.log(weights), indexing='ij')
    log_weights = sum(logs).ravel()
    return coordinates, log_weights


@lru_cache(maxsize=24)
def _haar_grid(radial, angles):
    if int(radial)!=radial or radial<2 or int(angles)!=angles or angles<4:
        raise ValueError('Global Haar quadrature needs radial>=2 and angles>=4')
    u,w=np.polynomial.legendre.leggauss(int(radial))
    u,w=(u+1.)/2.,w/2.
    angle=np.arange(int(angles))*2.*np.pi/int(angles)
    uu,aa,bb=np.meshgrid(u,angle,angle,indexing='ij')
    q=np.stack((np.sqrt(uu)*np.cos(aa),np.sqrt(uu)*np.sin(aa),
        np.sqrt(1.-uu)*np.cos(bb),np.sqrt(1.-uu)*np.sin(bb)),axis=-1).reshape(-1,4)
    weights=np.repeat(w,int(angles)**2)/int(angles)**2
    return q,np.log(weights)


def _global_orders(order,radial,angles):
    return ((int(order)+1)//2 if radial is None else int(radial),
            2*int(order)-2 if angles is None else int(angles))


def rotation_matrices(quaternions):
    q = np.asarray(quaternions, dtype=float)
    return np.einsum('...a,ijab,...b->...ij', q, ROTATION_FORMS, q, optimize=True)


class ExponentialAttitudeLaw:
    def __init__(self, A, Wg, Wb, g, b):
        self.A = _symmetric(A, (4, 4))
        self.Wg = _symmetric(Wg, (3, 3))
        self.Wb = _symmetric(Wb, (3, 3))
        self.g, self.b = np.asarray(g, float).copy(), np.asarray(b, float).copy()
        if self.g.shape != (3,) or self.b.shape != (3,) or not np.isfinite(self.g).all() or not np.isfinite(self.b).all():
            raise ValueError('World reference vectors must have three finite entries')
        self.gnorm, self.bnorm = float(np.linalg.norm(self.g)), float(np.linalg.norm(self.b))
        if self.gnorm == 0. or self.bnorm == 0.:
            raise ValueError('World reference vectors must be nonzero')
        self.Qg = np.einsum('i,ijab->jab', self.g, ROTATION_FORMS)
        self.Qb = np.einsum('i,ijab->jab', self.b, ROTATION_FORMS)
        self._Ag = self.A - np.trace(self.A) / 4. * np.eye(4)
        self._Wgg = self.Wg - np.trace(self.Wg) / 3. * np.eye(3)
        self._Wbg = self.Wb - np.trace(self.Wb) / 3. * np.eye(3)

    def _values(self, q):
        q = np.asarray(q, float)
        hg = np.einsum('...a,iab,...b->...i', q, self.Qg, q, optimize=True)
        hb = np.einsum('...a,iab,...b->...i', q, self.Qb, q, optimize=True)
        logp = np.einsum('...a,ab,...b->...', q, self.A, q, optimize=True)
        logp -= .5 * np.einsum('...i,ij,...j->...', hg, self.Wg, hg, optimize=True)
        logp -= .5 * np.einsum('...i,ij,...j->...', hb, self.Wb, hb, optimize=True)
        return logp, hg, hb

    def _relative_values(self, quaternions, reference, difference=None):
        # Differences of the quadratic forms are computed before subtracting
        # large log densities. Scalar gauges cancel identically on S3.
        q = np.asarray(quaternions, float)
        reference = _quaternion(reference)
        difference = q - reference if difference is None else difference
        total = 2. * reference + difference
        hg0 = np.einsum('a,iab,b->i', reference, self.Qg, reference)
        hb0 = np.einsum('a,iab,b->i', reference, self.Qb, reference)
        dg = np.einsum('...a,iab,...b->...i', difference, self.Qg, total, optimize=True)
        db = np.einsum('...a,iab,...b->...i', difference, self.Qb, total, optimize=True)
        value = np.einsum('...a,ab,...b->...', difference, self._Ag, total, optimize=True)
        value -= .5 * np.einsum('...i,ij,...j->...', dg, self._Wgg, 2.*hg0 + dg, optimize=True)
        value -= .5 * np.einsum('...i,ij,...j->...', db, self._Wbg, 2.*hb0 + db, optimize=True)
        return value, hg0 + dg, hb0 + db

    def relative_log_density(self, quaternions, reference):
        """Log density minus its reference value, with stable gauge cancellation."""
        return self._relative_values(quaternions, reference)[0]

    def log_density(self, quaternions):
        """Unnormalized log density on S3, retaining every quartic term."""
        return self._values(quaternions)[0]

    def features(self, quaternions, hg=None, hb=None):
        q = np.asarray(quaternions, float)
        if hg is None or hb is None:
            _, hg, hb = self._values(q)
        return np.concatenate((
            np.einsum('...a,kab,...b->...k', q, B4, q, optimize=True),
            -.5 * np.einsum('...a,kab,...b->...k', hg / self.gnorm, B3, hg / self.gnorm, optimize=True),
            -.5 * np.einsum('...a,kab,...b->...k', hb / self.bnorm, B3, hb / self.bnorm, optimize=True)), axis=-1)

    def parameters(self):
        return np.r_[np.einsum('ij,kij->k', self.A, B4),
                     self.gnorm**2 * np.einsum('ij,kij->k', self.Wg, B3),
                     self.bnorm**2 * np.einsum('ij,kij->k', self.Wb, B3)]

    def from_parameters(self, values):
        values = np.asarray(values, float)
        if values.shape != (19,) or not np.isfinite(values).all():
            raise ValueError('The trace-free law has 19 finite parameters')
        return ExponentialAttitudeLaw(np.einsum('k,kij->ij', values[:9], B4),
            np.einsum('k,kij->ij', values[9:14], B3) / self.gnorm**2,
            np.einsum('k,kij->ij', values[14:19], B3) / self.bnorm**2,
            self.g, self.b)

    def _derivatives(self, q):
        q = _quaternion(q)
        gradient = 2. * self._Ag @ q
        hessian = 2. * self._Ag.copy()
        for forms, weight in ((self.Qg, self._Wgg), (self.Qb, self._Wbg)):
            v = np.einsum('iab,b->ia', forms, q)
            h = v @ q
            wh = weight @ h
            gradient -= 2. * v.T @ wh
            hessian -= 2. * np.einsum('i,iab->ab', wh, forms) + 4. * v.T @ weight @ v
        return gradient, hessian

    def find_mode(self, reference, max_iterations=12):
        """Finite retracted Newton ascent, recharting after every accepted step.

        The unit projective trust radius bounds the numerical search step, not
        the attitude law or its uncertainty. Unlike an unbounded chart BFGS
        search, no trial can approach an infinite stereographic coordinate.
        """
        q = _quaternion(reference)
        iterations, evaluations = 0, 0
        damping_history = []
        reason = 'maximum_iterations_finite_result_retained'
        for iteration in range(int(max_iterations)):
            basis = projective_basis(q)
            gradient_q, hessian_q = self._derivatives(q)
            gradient = basis.T @ gradient_q
            information = -(basis.T @ hessian_q @ basis - float(q @ gradient_q)*np.eye(3))
            information = .5*(information+information.T)
            values, vectors = np.linalg.eigh(information)
            scale = float(np.max(np.abs(values)))
            # Damping stabilizes this optimizer solve only. It neither changes
            # A/W nor adds any state covariance/jitter to the probability law.
            damping = max(0.,-float(values[0])) + math.sqrt(np.finfo(float).eps)*scale
            if not np.any(gradient):
                if values[0] < 0.:
                    step = vectors[:,0].copy()  # escape an exact local minimum
                else:
                    reason = 'stationary_at_floating_precision'
                    break
            else:
                try:
                    step = np.linalg.solve(information+damping*np.eye(3),gradient)
                except np.linalg.LinAlgError:
                    step = gradient.copy()
                if not np.isfinite(step).all() or float(step@gradient) <= 0.:
                    step = gradient.copy()
            maximum = float(np.max(np.abs(step)))
            if maximum == 0.:
                reason = 'no_representable_ascent_step_finite_result_retained'
                break
            scaled_norm = float(np.linalg.norm(step/maximum))
            if maximum > 1./scaled_norm:
                step = (step/maximum)/scaled_norm
            if values[0]>0. and np.linalg.norm(step)<=8.*np.finfo(float).eps:
                reason='Newton_step_at_floating_coordinate_resolution'
                break
            slope = float(gradient@step)
            accepted = False
            for power in range(24):
                z = (2.**-power)*step
                radius2 = float(z@z)
                norm = math.sqrt(1.+radius2)
                difference = basis@z/norm-q*(radius2/(norm*(1.+norm)))
                trial = _quaternion(q+difference)
                if np.array_equal(trial,q):
                    continue
                improvement = float(self._relative_values(trial,q,difference=difference)[0])
                evaluations += 1
                if np.isfinite(improvement) and improvement >= 1e-4*(2.**-power)*slope:
                    q = trial
                    iterations += 1
                    damping_history.append(float(damping))
                    accepted = True
                    break
            if not accepted:
                reason = 'relative_density_no_longer_increases_finite_result_retained'
                break
        final_gradient = projective_basis(q).T@self._derivatives(q)[0]
        return q, {'optimizer': 'recharted_projective_Newton_trust_ascent',
            'optimizer_success': reason=='stationary_at_floating_precision',
            'message': reason, 'iterations': iterations,
            'maximum_iterations':int(max_iterations),
            'gradient_norm':float(np.linalg.norm(final_gradient)),
            'line_search_evaluations':evaluations,
            'projective_trust_radius':1.,
            'optimizer_damping_history':damping_history,
            'finite_result_used':True}

    def mode(self, reference, max_iterations=12):
        return self.find_mode(reference, max_iterations=max_iterations)[0]

    def find_modes(self,references,max_iterations=20):
        """Batched version of the same finite, recharted Newton ascent.

        Searches remain independent; vectorization changes only their cost.
        Each accepted step uses the exact quadratic/quartic relative density.
        """
        q=np.asarray(references,float).copy()
        if q.ndim!=2 or q.shape[1]!=4 or not np.isfinite(q).all():
            raise ValueError('Mode search references need finite shape (N,4)')
        norms=np.linalg.norm(q,axis=1)
        if np.any(norms==0.):raise ValueError('Mode search references must be nonzero')
        q/=norms[:,None]
        count=len(q);active=np.ones(count,dtype=bool);iterations=np.zeros(count,int)
        evaluations=np.zeros(count,int);reasons=['maximum_iterations_finite_result_retained']*count
        histories=[[] for _ in range(count)]

        def derivatives(points):
            gradient=2.*points@self._Ag.T
            hessian=np.broadcast_to(2.*self._Ag,(len(points),4,4)).copy()
            for forms,weight in ((self.Qg,self._Wgg),(self.Qb,self._Wbg)):
                v=np.einsum('iab,nb->nia',forms,points)
                h=np.einsum('nia,na->ni',v,points);wh=h@weight.T
                gradient-=2.*np.einsum('nia,ni->na',v,wh)
                hessian-=2.*np.einsum('ni,iab->nab',wh,forms)+4.*np.einsum('nia,ij,njb->nab',v,weight,v)
            return gradient,hessian

        def improvement(points,difference):
            total=2.*points+difference
            value=np.einsum('na,ab,nb->n',difference,self._Ag,total)
            for forms,weight in ((self.Qg,self._Wgg),(self.Qb,self._Wbg)):
                h0=np.einsum('na,iab,nb->ni',points,forms,points)
                dh=np.einsum('na,iab,nb->ni',difference,forms,total)
                value-=.5*np.einsum('ni,ij,nj->n',dh,weight,2.*h0+dh)
            return value

        for iteration in range(int(max_iterations)):
            ids=np.flatnonzero(active)
            if not len(ids):break
            points=q[ids];bases=_projective_bases(points)
            gradient_q,hessian_q=derivatives(points)
            gradient=np.einsum('nai,na->ni',bases,gradient_q)
            information=-(np.einsum('nai,nab,nbj->nij',bases,hessian_q,bases)
                -np.einsum('na,na->n',points,gradient_q)[:,None,None]*np.eye(3))
            information=.5*(information+information.transpose(0,2,1))
            values,vectors=np.linalg.eigh(information)
            scales=np.max(np.abs(values),axis=1)
            damping=np.maximum(0.,-values[:,0])+math.sqrt(np.finfo(float).eps)*scales
            transformed=np.einsum('nji,nj->ni',vectors,gradient)
            denominators=values+damping[:,None]
            quotient=np.divide(transformed,denominators,out=np.zeros_like(transformed),where=denominators!=0.)
            steps=np.einsum('nij,nj->ni',vectors,quotient)
            zero_gradient=~np.any(gradient,axis=1)
            escaping=zero_gradient&(values[:,0]<0.)
            steps[escaping]=vectors[escaping,:,0]
            stationary=zero_gradient&~escaping
            fallback=~np.isfinite(steps).all(axis=1)|(np.einsum('ni,ni->n',steps,gradient)<=0.)
            steps[fallback&~zero_gradient]=gradient[fallback&~zero_gradient]
            maximum=np.max(np.abs(steps),axis=1)
            normalized=np.divide(steps,maximum[:,None],out=np.zeros_like(steps),where=maximum[:,None]!=0.)
            normalized_norm=np.linalg.norm(normalized,axis=1)
            too_long=maximum*normalized_norm>1.
            steps[too_long]=normalized[too_long]/normalized_norm[too_long,None]
            at_resolution=(values[:,0]>0.)&(np.linalg.norm(steps,axis=1)<=8.*np.finfo(float).eps)
            stopped=stationary|(maximum==0.)|at_resolution
            for local in np.flatnonzero(stopped):
                index=int(ids[local]);active[index]=False
                reasons[index]=('stationary_at_floating_precision' if stationary[local]
                    else 'Newton_step_at_floating_coordinate_resolution')
            pending=~stopped
            slopes=np.einsum('ni,ni->n',gradient,steps)
            for power in range(24):
                loc=np.flatnonzero(pending)
                if not len(loc):break
                z=(2.**-power)*steps[loc];radius2=np.einsum('ni,ni->n',z,z)
                norm=np.sqrt(1.+radius2)
                difference=np.einsum('nai,ni->na',bases[loc],z)/norm[:,None]
                difference-=points[loc]*(radius2/(norm*(1.+norm)))[:,None]
                trial=points[loc]+difference
                trial/=np.linalg.norm(trial,axis=1)[:,None]
                gain=improvement(points[loc],difference)
                evaluations[ids[loc]]+=1
                changed=np.any(trial!=points[loc],axis=1)
                accepted=changed&np.isfinite(gain)&(gain>=1e-4*(2.**-power)*slopes[loc])
                accepted_loc=loc[accepted];accepted_ids=ids[accepted_loc]
                q[accepted_ids]=trial[accepted];iterations[accepted_ids]+=1
                pending[accepted_loc]=False
                for local,index in zip(accepted_loc,accepted_ids):histories[int(index)].append(float(damping[local]))
            for local in np.flatnonzero(pending):
                index=int(ids[local]);active[index]=False
                reasons[index]='relative_density_no_longer_increases_finite_result_retained'
        gradient_q,_=derivatives(q)
        final_gradient=np.einsum('nai,na->ni',_projective_bases(q),gradient_q)
        diagnostics=[{'optimizer':'batched_recharted_projective_Newton_trust_ascent',
            'optimizer_success':reasons[i]=='stationary_at_floating_precision','message':reasons[i],
            'iterations':int(iterations[i]),'maximum_iterations':int(max_iterations),
            'gradient_norm':float(np.linalg.norm(final_gradient[i])),
            'line_search_evaluations':int(evaluations[i]),'projective_trust_radius':1.,
            'optimizer_damping_history':histories[i],'finite_result_used':True} for i in range(count)]
        return q,diagnostics

    def proposal_covariance(self, reference, fallback=None):
        """Exact local log-density curvature used only to place integration nodes."""
        q = _quaternion(reference)
        basis = projective_basis(q)
        gradient, hessian = self._derivatives(q)
        information = -(basis.T @ hessian @ basis - float(q @ gradient) * np.eye(3))
        information = .5 * (information + information.T)
        try:
            factor = np.linalg.cholesky(information)
            inverse_factor = np.linalg.solve(factor, np.eye(3))
            value = inverse_factor.T @ inverse_factor
            return .5 * (value + value.T)
        except np.linalg.LinAlgError:
            if fallback is None:
                raise
            # This selects integration nodes only; no state variance is changed.
            fallback = _symmetric(fallback, (3, 3))
            np.linalg.cholesky(fallback)
            return fallback

    def integrate(self, reference, proposal_covariance, order=7, *,
                  global_radial=None,global_angles=None,local_mixture_weight=.999999,
                  mode_seeds=3,mode_iterations=20):
        """Adaptive deterministic multiple-chart/Haar importance quadrature.

        All found local charts contribute through one balance-mixture
        denominator. Mode searches only place integration nodes; their
        quaternion values do not replace or approximate the probability law.
        """
        reference=_quaternion(reference)
        covariance=_symmetric(proposal_covariance,(3,3))
        np.linalg.cholesky(covariance)
        centers=[(reference,covariance)]
        seed_q,_=_haar_grid(4,12)
        scores=self.relative_log_density(seed_q,reference)
        selected=np.argsort(scores)[-int(mode_seeds):][::-1] if mode_seeds else []
        spatial_seeds=_spatial_mode_seeds()
        search_seeds=np.concatenate((spatial_seeds,seed_q[selected]))
        modes,mode_diagnostics=self.find_modes(search_seeds,max_iterations=mode_iterations)
        center_diagnostics=[]
        merge_quaternion_distance=5e-7
        for search_index,(mode,diagnostic) in enumerate(zip(modes,mode_diagnostics)):
            if mode@reference<0.:mode=-mode
            entry={'seed_origin':('spatial_projective' if search_index<len(spatial_seeds) else 'highest_Haar_score'),
                'seed_index':(int(search_index) if search_index<len(spatial_seeds)
                    else int(selected[search_index-len(spatial_seeds)])), 'mode_optimizer':diagnostic}
            try:
                mode_covariance=self.proposal_covariance(mode)
                entry['proposal_available']=True
                # Only duplicate numerical proposals are coalesced. The
                # reference proposal is kept independently because its width
                # can differ from the mode curvature after process noise.
                duplicate=next((i for i,(old,_) in enumerate(centers[1:],1)
                    if np.linalg.norm(mode-np.copysign(1.,mode@old)*old)<=merge_quaternion_distance),None)
                if duplicate is None:
                    centers.append((mode,mode_covariance));entry['integration_chart']=len(centers)-1
                else:entry.update(integration_chart=duplicate,duplicate_integration_proposal=True)
            except np.linalg.LinAlgError:
                # Haar retains global support. This is an unavailable
                # numerical integration chart, never a discarded law branch.
                entry['proposal_available']=False
            center_diagnostics.append(entry)
        base,log_hermite=_hermite_grid(order)
        radial,angles=_global_orders(order,global_radial,global_angles)
        global_q,log_haar=_haar_grid(radial,angles)
        alpha=float(local_mixture_weight)
        if not 0.<alpha<1.:
            raise ValueError('The integration mixture weight must be strictly between zero and one')
        nodes,differences,rules=[],[],[]
        local_weight=alpha/len(centers)
        for center,chart_covariance in centers:
            lower=np.linalg.cholesky(chart_covariance)
            z=math.sqrt(2.)*base@lower.T
            radius2=np.einsum('ni,ni->n',z,z)
            scale=np.sqrt(1.+radius2)
            basis=projective_basis(center)
            q=(center+z@basis.T)/scale[:,None]
            local_difference=(z@basis.T)/scale[:,None]-center*(radius2/(scale*(1.+scale)))[:,None]
            center_dot=float(center@reference)
            if center_dot>0.:
                center_z=(projective_basis(reference).T@center)/center_dot
                center_radius2=float(center_z@center_z)
                center_scale=math.sqrt(1.+center_radius2)
                center_difference=(projective_basis(reference)@center_z)/center_scale-reference*(center_radius2/(center_scale*(1.+center_scale)))
            else:center_difference=center-reference
            nodes.append(q);differences.append(center_difference+local_difference)
            rules.append(math.log(local_weight)+log_hermite-1.5*math.log(math.pi))
        nodes.append(global_q);differences.append(global_q-reference)
        rules.append(math.log1p(-alpha)+log_haar)
        q=np.concatenate(nodes);difference=np.concatenate(differences)
        log_rule=np.concatenate(rules)
        densities=[np.full(len(q),math.log1p(-alpha)-math.log(2.*math.pi**2))]
        for center,chart_covariance in centers:
            lower=np.linalg.cholesky(chart_covariance)
            dots=q@center;tangent=q@projective_basis(center)
            valid=dots!=0.;z=np.zeros_like(tangent)
            with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
                z[valid]=tangent[valid]/dots[valid,None]
                white=np.linalg.solve(lower,z.T).T
                radius2=np.einsum('ni,ni->n',z,z)
                mahal=np.einsum('ni,ni->n',white,white)
                loglocal=(-math.log(2.)-1.5*math.log(2.*math.pi)
                    -float(np.log(np.diag(lower)).sum())-.5*mahal+2.*np.log1p(radius2))
            loglocal[~valid|~np.isfinite(loglocal)]=-np.inf
            densities.append(math.log(local_weight)+loglocal)
        log_mixture=logsumexp(np.stack(densities),axis=0)
        log_measure=log_rule-log_mixture
        relative_logp,hg,hb=self._relative_values(q,reference,difference=difference)
        relative_logz=float(logsumexp(relative_logp+log_measure))
        log_probability=relative_logp+log_measure-relative_logz
        weights=np.exp(log_probability);normalization=float(weights.sum())
        weights/=normalization;log_probability-=math.log(normalization)
        logz=float(self.log_density(reference))+relative_logz
        features=self.features(q,hg,hb);rotation=rotation_matrices(q)
        mean_rotation=np.einsum('n,nij->ij',weights,rotation)
        ehg,ehb=weights@hg,weights@hb
        centered_g,centered_b=hg-ehg,hb-ehb
        return {'qnodes':q,'weights':weights,'R':rotation,'hg':hg,'hb':hb,
            'Mqq':np.einsum('n,ni,nj->ij',weights,q,q),
            'Ngg':np.einsum('n,ni,nj->ij',weights,hg,hg),
            'Nbb':np.einsum('n,ni,nj->ij',weights,hb,hb),
            'Ehg':ehg,'Ehb':ehb,'mean_R':mean_rotation,
            'Cgg':(centered_g.T*weights)@centered_g,
            'Cbb':(centered_b.T*weights)@centered_b,
            'features':features,'feature_mean':weights@features,'logZ':logz,
            'relative_logZ':relative_logz,'log_measure':log_measure,'log_probability':log_probability,
            'reference':reference,'proposal_covariance':covariance,'order':int(order),
            'quadrature_nodes':len(q),'local_quadrature_nodes':len(centers)*len(base),
            'global_quadrature_nodes':len(global_q),'global_radial_order':radial,
            'global_angle_order':angles,'local_mixture_weight':alpha,
            'integration_centers':centers,'integration_center_diagnostics':center_diagnostics,
            'spatial_projective_mode_seeds':len(spatial_seeds),'top_Haar_mode_seeds':len(selected),
            'proposal_merge_quaternion_distance':merge_quaternion_distance,
            'integration_method':'deterministic_adaptive_multilocal_GH_and_global_Haar_MIS'}


def feature_target(Mq, Ng, Nb, g, b):
    g2, b2 = float(np.asarray(g) @ np.asarray(g)), float(np.asarray(b) @ np.asarray(b))
    return np.r_[np.einsum('kij,ij->k', B4, np.asarray(Mq)),
        -.5 / g2 * np.einsum('kij,ij->k', B3, np.asarray(Ng)),
        -.5 / b2 * np.einsum('kij,ij->k', B3, np.asarray(Nb))]


def forward_kl_projection(targetMq,targetNg,targetNb,initial_law,reference,
                          proposal_covariance,order=7,max_iterations=3,*,
                          global_radial=None,global_angles=None,local_mixture_weight=.999999):
    """Finite adaptive-normalizer forward-KL solve in all19 coefficients.

    Every candidate law gets its own deterministic multi-chart integration.
    Gradients/Hessians are moments of that same candidate law. This avoids
    treating one finite node support as the continuous S3 probability space.
    """
    reference=_quaternion(reference);basis=projective_basis(reference)
    target_matrix=_symmetric(targetMq,(4,4));denominator=float(reference@target_matrix@reference)
    if denominator<=0.:raise ValueError('Target quaternion moment has zero mass along the reference')
    target_proposal=basis.T@target_matrix@basis/denominator
    target_proposal=.5*(target_proposal+target_proposal.T)
    old_lower=np.linalg.cholesky(_symmetric(proposal_covariance,(3,3)))
    target_lower=np.linalg.cholesky(target_proposal)
    old_inverse=np.linalg.solve(old_lower,np.eye(3));target_inverse=np.linalg.solve(target_lower,np.eye(3))
    curvature_change=.5*basis@(old_inverse.T@old_inverse-target_inverse.T@target_inverse)@basis.T
    seed=ExponentialAttitudeLaw(initial_law.A+curvature_change,initial_law.Wg,initial_law.Wb,initial_law.g,initial_law.b)
    target=feature_target(targetMq,targetNg,targetNb,initial_law.g,initial_law.b)
    reference_feature=initial_law.features(reference)
    evaluations=0
    def evaluate(parameters):
        nonlocal evaluations
        law=initial_law.from_parameters(parameters)
        integral=law.integrate(reference,target_proposal,order=order,
            global_radial=global_radial,global_angles=global_angles,
            local_mixture_weight=local_mixture_weight)
        features=integral['features'];weights=integral['weights'];mean=weights@features
        gradient=mean-target;difference=features-mean
        hessian=(difference.T*weights)@difference
        value=integral['relative_logZ']+float(parameters@(reference_feature-target))
        evaluations+=1
        return value,gradient,.5*(hessian+hessian.T),law,integral
    original_parameters=initial_law.parameters();seed_parameters=seed.parameters()
    original_result=evaluate(original_parameters);seed_result=evaluate(seed_parameters)
    original_value,seed_value=original_result[0],seed_result[0]
    if original_value<seed_value:
        point=original_parameters.copy();result=original_result;selected='original_transported_coefficients'
    else:
        point=seed_parameters.copy();result=seed_result;selected='target_curvature_seed'
    value,gradient,hessian,law,integral=result
    diagnostics={'method':'finite_forward_KL_19_coefficient_adaptive_normalizer_damped_Newton',
        'maximum_iterations':int(max_iterations),'iterations':0,'all_features_retained':19,
        'quadrature_order':int(order),'finite_result_used':True,'damping_is_optimizer_only':True,
        'initialization':'lower_adaptive_dual_of_original_and_target_curvature_seed',
        'selected_initialization':selected,'initialization_dual_values':{
            'original_transported_coefficients':float(original_value),'target_curvature_seed':float(seed_value)},
        'target_moments_changed':False,'quartic_coefficients_seeded_unchanged':True,
        'proposal_is_exact_projective_covariance':False,
        'initial_moment_residual_definition':'selected numerical law before adaptive dual optimization',
        'initial_moment_residual':float(np.linalg.norm(gradient)),
        'integration_proposal_covariance':target_proposal.tolist(),
        'curvature_seed_coefficient_change_norm':float(np.linalg.norm(curvature_change)),
        'global_radial_order':integral['global_radial_order'],'global_angle_order':integral['global_angle_order'],
        'global_quadrature_nodes':integral['global_quadrature_nodes'],
        'local_mixture_weight':local_mixture_weight}
    starting_point=point.copy();scale=float(np.linalg.eigvalsh(hessian)[-1])
    damping=math.sqrt(np.finfo(float).eps)*scale;damping_history=[]
    reason='maximum_iterations_finite_result_retained'
    for iteration in range(int(max_iterations)):
        accepted=False
        for attempt in range(6):
            try:step=np.linalg.solve(hessian+damping*np.eye(19),-gradient)
            except np.linalg.LinAlgError:
                damping=damping*10. if damping>0. else np.finfo(float).eps
                continue
            if not np.isfinite(step).all():
                damping=damping*10. if damping>0. else np.finfo(float).eps
                continue
            for power in range(16):
                trial=point+(2.**-power)*step
                trial_result=evaluate(trial)
                if np.isfinite(trial_result[0]) and trial_result[0]<=value:
                    point=trial;value,gradient,hessian,law,integral=trial_result
                    accepted=True;diagnostics['iterations']+=1;damping_history.append(damping)
                    damping/=4.;break
            if accepted:break
            damping=damping*10. if damping>0. else np.finfo(float).eps
        if not accepted:
            reason='adaptive_dual_no_longer_decreases_finite_result_retained';break
    if not np.isfinite(point).all():raise FloatingPointError('Projection produced non-finite parameters')
    diagnostics.update(reason=reason,moment_residual=float(np.linalg.norm(gradient)),
        natural_gradient_norm=float(np.linalg.norm(gradient)),dual_value=float(value),
        coefficient_step_norm=float(np.linalg.norm(point-original_parameters)),
        optimizer_step_norm=float(np.linalg.norm(point-starting_point)),
        optimizer_damping_history=damping_history,input_moment_trace=float(np.trace(targetMq)),
        adaptive_normalizer_evaluations=evaluations,
        local_chart_count=len(integral['integration_centers']))
    return law,diagnostics
