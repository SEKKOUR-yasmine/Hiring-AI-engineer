# Full-time AI engineer test

The principal objective of this project is to evaluate the applicant's ability to learn new skills on the fly, build machine learning models in adherence to best practices and colaborate with others.

The applicant is also expected to write a modular code following good coding practices.

## Tasks

- **GaussianProcess.py**: Write a `GaussianProcess` class that embodies the Gaussian process regression model's functionality.
- **kernels.py**: Implement a selection of three kernel functions.
- **Kernel Operations**: Enable your kernels to perform addition (`+`) and multiplication (`*`) operations.
- **Fit the guassian process**: Fit your Gaussian process to the datasets provided and plot the results.
- **REST API via FastAPI**: Design a REST API using FastAPI to make your Gaussian process regression accessible over HTTP.
- **Dockerization**: Containerize your application with Docker, ensuring all dependencies are included for seamless setup and deployment.
- **Documentation**: Document your solution thoroughly with docstrings, inline comments, and a `readme.md` file detailing setup and usage.

## Setup

Clone the repository
```bash
git clone git@github.com:BIGmama-technology/Hiring-AI-engineer.git
```

Run `setup.sh`, this will create a virtual environment and install some dependencies
```bash
./scripts/setup.sh
```

Activate the virtual environment
```bash
source .venv/bin/activate
```

## Contribution guidelines

- design the structure of your repo in a modular way, example :

```
.
├── data
│   ├── international-airline-passengers.csv
│   └── mauna_loa_atmospheric_co2.csv
├── docs
│   └── report.pdf
├── LICENSE
├── output
│   └── figure_1.png
├── src
│   ├── __init__.py
│   ├── main.py
│   ├── data
│   │   └── data_loader.py
│   ├── models
│   │   ├── GaussianProcess.py
│   │   └── kernels.py
│   └── utils
│       └── utils.py
├── pyproject.toml
├── README.md
└── setup.cfg
```

- always use the virtual environment

```bash
# activate the virtual environment created by setup.sh
source .venv/bin/activate
```
- Make sure you include any requirements and dependencies in your `pyproject.toml` or `requirements.txt`.
- Type your code, document it and format it.
```python
# untyped, undocumented and unformatted code
import numpy as np
class gaussiankernel:
 def __init__(self,sigma=1.0):
  self.sigma=sigma
 def compute(self,x1,x2):
  return np.exp(-0.5 * np.linalg.norm(x1-x2)**2 / self.sigma**2)

```

```python
# typed, documented and formatted code
import numpy as np
from typing import Any, Union

class GaussianKernel:
    def __init__(self, sigma: float = 1.0) -> None:
        """
        Initialize the Gaussian kernel with a specified standard deviation (sigma).

        Parameters:
        sigma (float): The standard deviation of the Gaussian kernel.
        """
        self.sigma: float = sigma

    def compute(self, x1: Union[float, np.ndarray], x2: Union[float, np.ndarray]) -> Any:
        """
        Compute the Gaussian kernel between two points.

        Parameters:
        x1 (Union[float, np.ndarray]): The first point or vector.
        x2 (Union[float, np.ndarray]): The second point or vector.

        Returns:
        The computed Gaussian kernel value.
        """
        return np.exp(-0.5 * np.linalg.norm(x1 - x2) ** 2 / self.sigma ** 2)

```

- Commit often and write meaningful commit messages.
- Create a new branch with your name, push your code to it and create a pull request once you finish your contribution.

## Resources

Candidates should leverage the following resources for guidance:

- [Good practices](https://goodresearch.dev/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Intro to Docker](https://docker-curriculum.com/)
- [What are gaussian processes : interactive guide](https://distill.pub/2019/visual-exploration-gaussian-processes/)
- [Kernel cookbook](https://www.cs.toronto.edu/~duvenaud/cookbook/)
- [Packaging with pip](https://packaging.python.org/en/latest/tutorials/packaging-projects/)

btawfiq inchalah
