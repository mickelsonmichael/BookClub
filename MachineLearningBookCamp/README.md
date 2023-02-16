# Machine Learning Bookcamp

## Dockerfile

The [Dockerfile](./Dockerfile) here can be used to easily spin up an environment based on the instructions within Appendix A. It will perform the following actions:

- Update jupyter
- Create the `~/notebooks` directory
- Update the kaggle CLI
- Copy `kaggle.json` to the proper location
- Clones the source code into `~/mlbookcamp-code`

The only manual step required is to create a Kaggle account and then create an access token and put the resulting `kaggle.json` file here, next to the Dockerfile.
See [Kaggle's documentation](https://www.kaggle.com/docs/api#authentication) or Appendix A for information on how to do this.

Then, you can simply perform the following commands:

```shell
$> docker build --tag mlbc .

$> docker run -it mlbc
```
