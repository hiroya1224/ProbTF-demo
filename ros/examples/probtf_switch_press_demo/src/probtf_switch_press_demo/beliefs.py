"""Frame names shared by native transform-law propagation and publication."""

BODY_FRAMES = tuple("link_{}".format(index) for index in range(1, 7)) + ("tool0", "rod_tip")
