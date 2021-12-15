Follow the instructions in README.md to install dependency

To train a policy
./train $ROBOT $GAIT $POLICY_FOLDERNAME

For example, to train a trotting policy for mini cheetah
./train MC trot trot01

This would create a folder named trot01 under the directory motion_imitation/data/policies/MC and save the trained policy there