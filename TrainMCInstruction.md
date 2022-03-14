Follow the instructions in README.md to install dependency

Note: Use Python 3.6.8. This is compatible with the tensorflow version specified in requirements.txt
``` git checkout dev-heli ```

### Train
To train a policy
```./train $ROBOT $GAIT $POLICY_FOLDERNAME```

For example, to train a trotting policy for mini cheetah
```./train MC trot trot01```

This would create a folder named trot01 under the directory motion_imitation/data/policies/MC and save the trained policy there

### Test
There are a couple of things to do to run a policy from a constant static pose

Comment line 204 ```self._sync_sim_model(perturb_state) ``` in file ```imitation_task.py```

Set ```ref_state_init_perturb = 0``` and ```enable_rand_init_time=False``` at line 117 in file ```env_builder.py```

If need to run Wenhao's fine-tuned A1 trotting policy, then set ```warmup_time=0.75``` at line 117 in file ```env_builder.py```




