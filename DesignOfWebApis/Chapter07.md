# 7 Designing a concise and well-organized API

- "Less is more" and "a place for everything and everything in its place"

## Chapter Summary

- Organize properties by sorting, naming consitently, or grouping them in related sub-groups
- Use appropriate HTTP response codes, add categories to your error messages, and sort them by their importance
- Focus on grouping goals by functionality
- Keep the number of properties around 20 or fewer
- Keep the depth at or below 3 levels
- "Avoid creating _does-it-all_ goals
- Utilize granularity to split goals up into smaller functional endpoints

## 7.1 Organizing an API

- You should group "buttons" by their function and organize them to make them more usable
  - e.g. group the numbers together and sort them, put the power button near the top so it cannot be accidentally pressed during usage

## 7.1.1 Organizing data

- Similarly, data should be grouped together in an object
- Related concepts should be similarly named and located close together in the data structure
- For example, the book publisher should be near the published date, or possibly even grouped together in a nested structure

Example with the publisher properties located spacially close to one another, with both having the word "publish" in the name.

```json
{
  "title": "The Design of Web APIs",
  "author": "Arnaud Lauret",
  "publisher": "Manning Publications",
  "publishedDate": 2012
}
```

Example with the publisher properties located within the same sub-property. When using this method, it also makes it easier to enforce requirements for related data; if the publisher isn't a required field, but both the date and name of the publisher are required if either is present, then having them in a sub-group simplifies that validation logic via a JSON Schema.

```json
{
  "title": "The Design of Web APIs",
  "author": "Arnaud Lauret",
  "publication": {
    "name": "Manning Publications",
    "date": 2012
  }
}
```

### 7.1.2 Organizing feedback

- When you receive a status code that you don't understand in you application, like a `413`, then you should just treat it as if it were the base status code for that _class_, in this case a simple `400` error.
- If you return custom errors from your application, they should be sorted from *most* to *least* critical

### 7.1.3 Organizing goals

Using the OpenAPI specification, you can group related functions together using a `tags` property. This allows you to visually group related processes together when the specification is displayed in a GUI.
For instance, you may give the `/purchase` and `/rent` endpoints of a book API the same `transactions` tag so they can be closely related by their core concepts to make it easier for users to discover.

It is also beneficial to the users to sort the categories (see `tags`) by the common usage.
Odds are good that a user may want to look up books before they perform a transaction with one, so the `books` tag should come before the `transactions` tag in the GUI and specification.

Finally, it is good practice to sort your HTTP verbs in a predictable manner. If you commonly have a `GET`, `POST`, and `DELETE` verb for several endpoints, then be sure to put them in the same, predictable order each time.

#### Exercise

How would you organize the following goals?

- `DELETE /images/{imageId}` | Removes an image from the current user's feed
- `GET /images` | Gets current user's feed
- `GET /users/{userId}/images` | Gets another user's feed
- `GET /users/{userId}` | Gets another user's information
- `POST /images` | Adds image to current user's feed
- `GET /me` | Gets the current user's information

## 7.2 Sizing an API

You don't want your API to be so large that it becomes unwieldy and difficult to use. Keep your inputs as small as possible and your outputs as small as convenient.

### 7.2.1 Choosing data granularity

For inputs, there should be the *least possible number of properties* for the endpoint to function, with a maximum nested depth of around 3.

For outputs, there should be as *as many properties as functionally relevant* with a recommended maximum of about 20 properties and a maximum depth of around 3.

In both input and output, however, it's important to choose limits that make sense for your domain.

### 7.2.2 Choosing goal granularity

At times, you may need to update nested data, but you need to be careful and allow for granularity in your endpoints.
Take the data below, for example:

```json
{
  "title": "The Design of Web APIs",
  "ratings": [
    { "username": "mike", "rating": 4 },
    { "username": "graham", "rating": 3 },
  ]
}
```

You should not need to update the entire book in order to update a single user's rating, that should be its own endpoint.
Additionally, you should not need to retrieve the entire list of ratings when pulling a book down, that also should be a seperate endpoint; the ratings could be a large collection of large items and would be come a burdon on the network and the user.

#### Exercise

How would you organize and split this shopping API goals list?

| Goal | Category | API |
| ---- | -------- | --- |
| Create user | USERS | `PUT /user` |
| Search for products | | | 
| Get product's information | | |
| Add product to shopping cart | | |
| Remove product from cart | | |
| Check out cart | | |
| Get cart detail | | |
| List orders | | |
| Add produt to catalog | | |
| Update a product | | |
| Replace a product | | |
| Delete a product | | |
| Get an order's status | | |
| Update user | | |
| Delete user | | |

