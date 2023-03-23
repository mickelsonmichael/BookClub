# Machine Learning Bookcamp

## Dockerfile

The [Dockerfile](./Dockerfile) here can be used to easily spin up an environment based on the instructions within Appendix A. It will perform the following actions:

- Installs necessary python dependencies
- Create the `/mlbc/notebooks` directory
- Update the kaggle CLI
- Copy `kaggle.json` to the proper location
- Clones the source code into `/mlbc/mlbookcamp-code`
- Creates a `launch-notebook` command

The only manual step required is to create a Kaggle account and then create an access token and put the resulting `kaggle.json` file here, next to the Dockerfile.
See [Kaggle's documentation](https://www.kaggle.com/docs/api#authentication) or Appendix A for information on how to do this.

Then, you can simply perform the following commands:

```shell
$> docker build --tag mlbc .

$> docker run -p 8888:8888 -it -v /some/local/path:/mlbc/notebooks mlbc
```

> NOTE: The `-v` option is required if you want your progress to be saved.

The container will automatically launch the Jupyter notebooks command when run.
To skip this, simply add `/bin/bash` to the end of your `docker run` command to specify the entry command (e.g., `docker run mlbc /bin/bash`).

### Installing Python Packages

You can add new dependencies in two ways.
If you plan to check the dependencies in to the repository, add them to the [requirements.txt](./requirements.txt) file on a new line.

If you want to just install them in the Jupyter notebook, you can simply use `!pip install <package>` inside a Jupyter code block.

For example, to install pandas:

```jupyter
!pip install pandas
```

Then press `Ctrl+Enter` to execute the code block.

### Using datasets

Data sets (if pre-installed) should be available in the `/mlbc` directory.
For example, to import the CSV from chapter 6, rather than using

```python
# incorrect
df = pd.read_csv("CreditScoring.csv")
```

You must prepend the `/mlbc/` directory to the path

```python
# correct
df = pd.read_csv("/mlbc/CreditScoring.csv")
```
