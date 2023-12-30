# xai_framework

### Project Structure

``` bash
xai_framework/
│
├── datasets/ # Datasets folder
├── notebooks/ # Jupyter notebooks
│   └── example.ipynb # Example notebook that uses the framework
├── xai_framework/ # Code of the library
│   ├── methods/
│   │   └── shap_method.py # Example method implementation
│   ├── types.py # Common wrappers and types
│   └── utils.py # Utility functions
│
├── .default.env # Default environment variables
├── .envrc # automatically loads environment variables from .default.env and .env with direnv
├── .dev_requirements.txt # development requirements (jupyter, etc.)
├── .requirements.txt # framework requirements (torch, etc.)
└── .env # environment variables ignored by git (optional .default.env override)
```
### Development Instructions

#### Setting up the environment

Project related commands are mostly inside the `Makefile`. To display the commands run:

``` bash
make help
```

For `development` run the following commands:

``` bash
make env # makes the python virtual environment
make requirements_dev # installs the dependencies inside the environment
```

This creates the virtual environment which can then be used for the Jupyter notebooks and terminal usage. If you run python scripts inside the terminal you have to activate the environment with:

``` bash
source ./venv/bin/activate
```

#### Optional

Install `direnv`. This library is used to load environment variables from `.envrc` file upon folder entry inside the terminal. It declutters the `bash` profile and allows to keep environment variables in one place. After installing `direnv` check the setup instructions [here](https://direnv.net/docs/hook.html). Each modification of `.envrc` file requires to run `direnv allow` command to reload the environment variables.

#### OSX

Install `gawk` as OSX `awk` differs from GNU `awk` as it is used inside the `Makefile`.

``` bash
brew install gawk 
```
