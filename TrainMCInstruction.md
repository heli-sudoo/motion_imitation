Follow the instructions in README.md to install dependency

Note: Use Python 3.6.8. This is compatible with the tensorflow version specified in requirements.txt
``` git checkout dev-heli ```

To train a policy
```./train $ROBOT $GAIT $POLICY_FOLDERNAME```

For example, to train a trotting policy for mini cheetah
```./train MC trot trot01```

This would create a folder named trot01 under the directory motion_imitation/data/policies/MC and save the trained policy there