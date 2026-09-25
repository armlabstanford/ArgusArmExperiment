Instructions for running the robot arm trajectories.

## Setup
```bash
conda activate yam_arms
```

Make sure the arm is powered on (power brick light is lit and E-stop is open)!

Bring up the CAN interface before running any arm commands:

```bash
sudo ip link set can0 up type can bitrate 1000000
```

To verify:

```bash
ip link show can0
# Should show state UP
```

If the adapter is unresponsive, reset it:

```bash
sudo bash robots_realtime/dependencies/i2rt/scripts/reset_all_can.sh
```

Now, ensure the ARGUS and MoCap data collections are running.

## Run trial
# Quick tuning the trajectories:
```bash
bash argus_experiment/trajectories/run_trajectories.sh TRIAL_NAME
```

# RUN THESE TRAJECTORIES

## Linear sawtooth (x, y, z sweeps; constant speed in m/s)
```bash
bash argus_experiment/trajectories/run_linsawtooth_v0.1.sh linear-sawtooth-0.1
```

```bash
bash argus_experiment/trajectories/run_linsawtooth_v0.5.sh linear-sawtooth-0.5
```

```bash
bash argus_experiment/trajectories/run_linsawtooth_v1.0.sh linear-sawtooth-1.0
```

## Linear sinusoid (x, y, z sweeps; peak speed in m/s)
```bash
bash argus_experiment/trajectories/run_linsinusoid_v0.1.sh linear-sinusoid-0.1
```

```bash
bash argus_experiment/trajectories/run_linsinusoid_v0.5.sh linear-sinusoid-0.5
```

```bash
bash argus_experiment/trajectories/run_linsinusoid_v1.0.sh linear-sinusoid-1.0
```

## Angular sawtooth (roll, pitch, yaw sweeps; constant speed in rad/s)
```bash
bash argus_experiment/trajectories/run_angsawtooth_v0.2.sh angular-sawtooth-0.2
```

```bash
bash argus_experiment/trajectories/run_angsawtooth_v1.0.sh angular-sawtooth-1.0
```

```bash
bash argus_experiment/trajectories/run_angsawtooth_v2.0.sh angular-sawtooth-2.0
```

## Angular sinusoid (roll, pitch, yaw sweeps; peak speed in rad/s)
```bash
bash argus_experiment/trajectories/run_angsinusoid_v0.2.sh angular-sinusoid-0.2
```

```bash
bash argus_experiment/trajectories/run_angsinusoid_v1.0.sh angular-sinusoid-1.0
```

```bash
bash argus_experiment/trajectories/run_angsinusoid_v2.0.sh angular-sinusoid-2.0
```

Trial ends automatically. End the ARGUS and MoCap data collection.

Trial name, parameters and arm information should be saved under ~/ArgusArmExperiment/argus_experiment/trajectories/recordings