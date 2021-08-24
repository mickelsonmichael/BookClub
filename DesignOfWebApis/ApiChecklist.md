# The API Checklist

Whether you're designing, writing, or reviewing an API, it's always helpful to have a list of "to do" items to ensure you follow best practices and procedures.
This document includes some of the common practices and methods you should consider when writing a good API.
Feel free to add your own insights and checks to the document, but try to keep the items broad enough that the check list doesn't become a chore, and instead remains a tool.

For some additional insight, check out the [Azure Architecture Web API best practices](https://docs.microsoft.com/en-us/azure/architecture/best-practices/api-design) document.

## Predictability

- [ ] All data types conform to a single format (e.g. dates all ISO format strings, account numbers all padded with leading zeroes)
- [ ] Property names for data types are consistent (e.g. dates follow the format `<target>Date`, `balanceDate`, `creationDate`)
- [ ] Common properties should have consistent naming (e.g. if account numbers are consistently used, they should all share the same `accountName` scheme)
- [ ] Each level of a URL should always have the same meaning (e.g `<action>/<id>`, `delayed-transfers/{transferId}` not `transfers/delayed/{transferId}`)
- [ ] Each level of a URL should have the same plurality (e.g. `users/{userId}`, `accounts/{accountId}` and not `account/{accountId}`
- [ ] Keep verbs consistent (e.g. always use `get` or `read` never both)
- [ ] If you don't utilize additional codes beyond `200` (e.g. `201 Created` or `202 Accepted`), don't introduce them for new endpoints
- [ ] If you use generic codes (e.g. `MISSING_REQUIRED_FIELD`) then use them across the entire API
- [ ] Look up the "standards" for your domain/endpoint and ensure you follow them (e.g. look up `<domain> standards` or `<domain> format` in a search engine)
- [ ] If users may want data in different formats, provide them (e.g. `appliction/json`, `test/csv`)
- [ ] If users may want data in different languages, provide them (e.g `en-US`, `fr-FR`)
- [ ] When using paging and filtering, the current page should be a property called `page`
- [ ] When using paging and filtering, the requested page size should be a property called `pageSize`
- [ ] When using paging and filtering, consider utilizing the `Range` header to specify the range of items instead of `page` and `pageSize`
- [ ] When using paging, provide paging metadata to the users (e.g. `page`, `totalPages`, `nextPageUrl`)
- [ ] Consider providing a `_links` property with helpful navigations to related endpoints
- [ ] Provide an `OPTIONS` response to list the valid `HTTP` verbs available at an endpoint


