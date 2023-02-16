# Machine Learning Bookcamp

## Dockerfile

The [Dockerfile](./Dockerfile) here can be used to easily spin up an environment based on the instructions within Appendix A. It will perform the following actions:

- Update jupyter
- Create the `~/notebooks` directory
- Update the kaggle CLI
- Copy `kaggle.json` to the proper location
- Clones the source code into `~/mlbookcamp-code`
- Creates a `launch-notebook` command

The only manual step required is to create a Kaggle account and then create an access token and put the resulting `kaggle.json` file here, next to the Dockerfile.
See [Kaggle's documentation](https://www.kaggle.com/docs/api#authentication) or Appendix A for information on how to do this.

Then, you can simply perform the following commands:

```shell
$> docker build --tag mlbc .

$> docker run -p 8888:8888 -it mlbc
```

To launch the Jupyter notebooks (at `/mlbc/notebooks`) you can use the `launch-notebook` command.
