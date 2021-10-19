# API Security in Action

## Using the Docker Image

If you want to avoid installing Java and Maven, you can utilize this helpful docker image.
This will **require docker to be installed**, but as developers I hope that's a common enough installation.

Simply build the image

```bash
cd ApiSecurityInAction
docker build . --tag api_sec
```

Then run the image, either in the foreground with `-it` or as a daemon via `-d`, and set the `CHAPTER` environmental variable to the name of the branch you want to run (excluding feature/oauth2). If you don't set the `CHAPTER` variable, it will default to `chapter03-end`.

```bash
docker run --rm -it -p 4567:4567 --env CHAPTER=chapter03-end api_sec
```

Once the container is running, you should be able to make requests through Postman as long as you turn off SSL Verification. Postman will fail the requests and display a popup with a link to easily do this.

If you are looking to run your own code in the container, you can overwrite the entry command and provide a [shared volume](https://docs.docker.com/storage/volumes/) pointing to your application code. See the below example:

```bash
docker run --rm -it -p 4567:4567 --mount source=/path/to/code,destination=/src/my_code api_sec bash
```
