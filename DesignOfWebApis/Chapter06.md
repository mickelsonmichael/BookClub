# Designing a predictable API

__Design an API that's intuitive and discoverable.__

## 6.1 Being Consistent

- Play and Pause have standard universally recognized icons. A user can predict the function of these buttons without other knowledge.
- Using standard conventions can allow an API to be predictable as well.

### 6.1.1 Designing consistent data

#### Naming

| Command | Inconsistent | Consistent|
| --- | --- | ---|
| get Accounts | accountNumber | accountNumber |
| get Account | number | accountNumber |
| transfer money | source | sourceAccountNumber |

- Use the same name for the same data across the API.
- Use similar variations to provide additional information.

| Command | Inconsistent | Consistent|
| --- | --- | ---|
| get Accounts | balanceDate | balanceDate |
| get Account | dateOfCreation | creationDate |
| transfer money | executionDay | executionDate |

- Even with unrelated data follow the same convention to provide additional information.

#### Types
  
| Command | Property | Inconsistent | Consistent|
| --- | --- | ---| --- |
| get Accounts | accountNumber | "0001234567" (string) | "0001234567" (string) |
| get Account | accountNumber | "001234567" (string)  | "0001234567" (string) |
| transfer money | sourceAccountNumber | 1234567 (int) | "0001234567" (string) |

- Use a standard representation for a the same data and same concept.
- Once a user sees a specific representation stick with it.

| Command | Property | Inconsistent | Consistent|
| --- | --- | ---| --- |
| get Accounts | balanceDate | "2018-03-23" (string) | "2018-03-23" (string) |
| get Account | creationDate | 1423267200 (int)  | "2015-02-07" (string) |
| list transactions | executionDate | "2018-23-03" (int) | "2018-03-23"  (string) |

- Seek uniformity. Having many different date formats would confuse consumers.

#### API URL

- `/accounts/{accountNumber}` vs `/transfer/{transferId}`. If you use plurals for a set then stick with it.
- If the standard is `{collection}/{resourceID}`, don't introduce unexpected levels like `/transfers/delayed/{transferId}`. An improved example `/delayed-transfers/{transferId}`.
  
#### Date Representation

```json
Return array data in a property called items
{
  "items": [
    { "accountNumber": "000123456" }
  ]
}
```

```json
Return array directly
[
  {"id": "123-567"}
]
```

- Stick with standard data representation.
- If you return a set of data in a property called items, stick with that and don't return an array directly.
- Consistency

### 6.1.2 Designing Consistent goals

- "read account" vs "get user information". The same action should use the same verb.
- REST APIs make the programmatic representation more consistent. `GET /resource-path`
- Use consistent names, data types, formats, and organization for all inputs.
- The same applies to return codes. If you always use `200 OK` when creating something don't introduce `201 Created` or `202 Accepted`.

### 6.1.3 The four levels of consistency

### 6.1.4 Copying others: Following common practices and meeting standards

### 6.1.5 Being consistent is hard and must be done wisely

## 6.2 Being adaptable