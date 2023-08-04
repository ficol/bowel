Project structure
------------

    │
    ├── README.md           <- this Readme
    │
    ├── environment.yml     <- the environment file to install project dependencies
    │
    ├── data
    │     ├── interim         <- intermediate data that has been transformed
    │     ├── processed       <- the final, canonical data sets for modeling
    │     └── raw             <- the original, immutable data dump
    │
    ├── docs                <- project documentation
    │
    ├── models              <- trained and serialized models, model predictions and/or model summaries
    │
    ├── notebooks           <- Jupyter notebooks
    │
    ├── references          <- data dictionaries, manuals, and all other explanatory materials
    │
    ├── reports             <- generated analyses as HTML, PDF, LaTeX, etc.
    │
    ├── tests               <- Unit and model tests.
    │
    └── bowel                      <- project source code
          │
          ├── config.py              <- configs and constants
          │
          ├── data                   <- modules to process and load data
          │
          ├── models                 <- modules to train models and make predictions
          │
          ├── utils                  <- modules with project-wide utilities
          │
          └── visualization          <- modules to create results-oriented visualizations

--------

# Setup

Create environment using conda:

```

$ conda env create -f environment.yml

```

To preprocess and train model:

```

$ conda activate bowel

$ make clean # remove files created during preprocessing data

$ make data # generate data

$ python -m bowel.models.train train --model <path to save model> --config <path to config file> # train model

```

To get metrics on trained model:

```

$ python -m bowel.models.train test --model <path to load model>

```

To evaluate:

```

$ python -m bowel.models.predict --model <path to load model> --wav_file <path to wav file> --csv_output <path to csv file to save result>

```

To generate raport:

```

$ python -m bowel.models.statistics <path to csv file with sounds> <path to save xlsx file> --wav_file <path to wav_file>

```

To visualize predictions:

```

$ python -m bowel.visualization.audio_plot <path to wav file> --config <path to config file with spectrogram parameters> --predict_csv <path to csv file with predicted annotations> --truth_csv <path to csv file with true annotations> -o <offset> -d <duration>

```

---
