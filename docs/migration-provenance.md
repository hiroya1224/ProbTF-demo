# Migration provenance

The integration branch preserves both source repositories as non-squashed
subtree merge parents. The source worktrees remain at their original paths and
were not modified during this migration.

| Source | Source HEAD | Root commit | Commits | Import commit | Imported path | Tree |
| --- | --- | --- | ---: | --- | --- | --- |
| `probik_demo` | `c06a2eabc258fbf55fe167f71a0253afa6b24e0b` | `9f9dedfd68dd8cac61d8f94346f2893386c93fe8` | 9 | `0d6bfa0f4bc729a6f1c4bf97fe8c90bf90eead51` | `ros/symaware_grasp` | `c27f4a77b2b208947974b3e3b141872d5a0d43e7` |
| `deflecomp` | `1508cac426b68d2e08546243223cf90ac200b7ce` | `a770384cf0e1a639a20068da5ae28673b9204bea` | 25 | `a2e8429cb56055bf7c8afac6b5b47a6aa8b9ba49` | `ros/deflecomp` | `eabea8567f87ccdf40572d548bd09695afc0fe68` |

At each import commit, the imported subtree has exactly the same Git tree ID as
the corresponding source HEAD. This covers all tracked source files: 56 from
`probik_demo` and 88 from `deflecomp`.

Run the repository-local audit from any directory:

```bash
python3 tools/verify_source_history.py
```

## Current mapping

- Reusable Python modules moved to `src/` while retaining their established
  import namespaces.
- The former `probik_demo` ROS package is now `ros/symaware_grasp`.
- Custom messages moved to `ros/probik_msgs`.
- Deflection ROS packages, documentation, configuration, launch files, URDF,
  and RViz assets remain under `ros/deflecomp`.
- ProbIK lectures, plans, configuration, launch files, URDF, and RViz assets
  remain under `ros/symaware_grasp`.
- BinghamNLL is pinned at `4cfae2a3bac12c4ecb3bd563b5efe93a1d8c3a78`
  from its `develop` branch in `third_party/BinghamNLL`.

The original repository state can be inspected directly with `git show` using
either source HEAD above, independent of the post-import moves and renames.

