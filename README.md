# AM01 - Detection of Anomalous Behaviour in Industrial Robot

## Project Information
- **Project Code**: 2026/AM01
- **Project Owner**: Alessio Mascolini - alessio.mascolini@polito.it
- **Title**: Detection of Anomalous Behaviour in Industrial Robot

## Project Overview
This project involves implementing an adversarial autoencoder (AAE) for anomaly detection on a Kuka industrial robot dataset. The dataset consists of time-series data collected from the robot's various sensors, including joint angle positions, velocity, current and power usage values.

The problem to solve is understanding when, due to an error in configuration or aging, the robot is moving more slowly or less precisely than normal. The AAE model will be trained to learn the normal movement patterns of the robot using the training data. Once the model has learned these patterns, it should be able to identify any deviations from them and flag them as anomalies.

The aim of the project is to evaluate whether an adversarial component improves the performance of a traditional autoencoder to detect anomalies. The choice of appropriate metrics to evaluate your result will be part of the examination.

## Applications
- Industrial, Anomaly Detection

## References
- Kim S. et al., "Towards a Rigorous Evaluation of Time-series Anomaly Detection", 2022

## Challenges
- Using adversarial training
- Handling time series data
- Comparing models

## Project Structure
```
AM01-Detection-of-Anomalous-Behaviour-in-Industrial-Robot/
├── data/
│   ├── raw/
│   │   └── KukaVelocityDataset/
│   └── processed/
├── src/
│   ├── data/
│   ├── models/
│   ├── utils/
│   └── main.py
├── notebooks/
├── tests/
├── config/
├── reports/
├── docs/
├── requirements.txt
├── setup.py
└── main.py
```

## Dataset Description

### Files
| File | Shape | Description |
|------|-------|-------------|
| KukaNormal.npy | (233,792, 86) | Normal robot movement data |
| KukaSlow.npy | (41,538, 87) | Anomalous/slow movement data |
| KukaColumnNames.npy | (87,) | Feature column names |

### Features
- **Power metrics**: apparent_power, current, frequency, phase_angle, power, power_factor, reactive_power, voltage
- **Sensors**: Accelerometer (AccX/Y/Z), Gyroscope (GyroX/Y/Z), Joint angles (q1-q4), Temperature (temp)

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python src/main.py
```

## TODO
- [ ] Data exploration and preprocessing
- [ ] Baseline autoencoder implementation
- [ ] Adversarial autoencoder implementation
- [ ] Model training and evaluation
- [ ] Results comparison
- [ ] Final reporting

## License
Academic project for Machine Learning in Applications course - Politecnico di Torino