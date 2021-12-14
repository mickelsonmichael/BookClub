# API Security in Action

## Contents

Not all chapters have notes currently, but those that do will be linked here for ease of access, or available by browsing through the files.

- [Chapter 04](https://github.com/mickelsonmichael/BookClub/tree/main/ApiSecurityInAction/chapter04.md)
- [Chapter 08](https://github.com/mickelsonmichael/BookClub/tree/main/ApiSecurityInAction/chapter08.md)
- [Chapter 09](https://github.com/mickelsonmichael/BookClub/tree/main/ApiSecurityInAction/chapter09.md)

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
