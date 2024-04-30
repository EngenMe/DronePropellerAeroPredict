# DronePropellerAeroPredict

DronePropellerAeroPredict is a project aimed at predicting the aerodynamic performance of drone propellers using machine learning techniques. This project utilizes datasets containing geometric and performance data of drone propellers to train predictive models.

## Introduction

DronePropellerAeroPredict is designed to assist drone enthusiasts, engineers, and researchers in predicting key aerodynamic parameters such as thrust coefficient (CT), power coefficient (CP), and efficiency (eta) for various drone propellers. By leveraging machine learning algorithms, this project provides accurate predictions based on input geometrical features of the propellers.

## Features

- **Data Loading:** DataLoader module loads and preprocesses geometric and performance data of drone propellers.
- **Model Training:** RandomForestReg module trains random forest regression models for predicting CT and CP.
- **Model Evaluation:** Results module evaluates model performance and provides error metrics.
- **Visualization:** Results module visualizes predicted results compared to actual values for a randomly selected propeller.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/EngenMe/DronePropellerAeroPredict.git
```

2. Install the required packages:

pip install -r requirements.txt

## Usage

1. Navigate to the project directory:

```bash
cd DronePropellerAeroPredict
```

2. Run the main script:

```bash
python main.py
```

3. Follow the on-screen prompts to build the model or visualize results.

## Contributing

We welcome contributions from the community! Please refer to the [CONTRIBUTING.md] file for more information on how to contribute to this project.

## Authors

* **Mohamed Farouk Hasnaoui** - *DronePropellerAeroPredict* - [EngenMe](https://github.com/EngenMe)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

This project benefited from the insights and support of several distinguished professionals:

- **Prof. Naeem Ramzan**, Head of the Engineering and Computing School at the University of the West of Scotland.
- **Dr. Djamalddine Boumezerane**, Associate Professor in the Civil Engineering Department at the University of the West of Scotland.
- **Prof. Arezki Smaili**, Head of the Laboratory of Green and Mechanical Engineering Department at the National Polytechnic School of Algiers.
- **Dr. Abdelhamid Bouhelal**, Associate Professor at the National Polytechnic School of Algiers.
- All members of the Computing School research team at the University of the West of Scotland for their invaluable contributions and feedback.

Special thanks to these individuals and teams for their guidance and encouragement throughout the development of this project.
