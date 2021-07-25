# Describing an API with an API description format
## 4.1 What is an Api description format?
- A data format used to describe the API
- OAS – OpenAPI Specification
  - Language agnostic format for REST API
  - Open source
  - Formerly known as the Swagger Specification
  - Basic Structure: https://swagger.io/docs/specification/basic-structure/
  - Can be written in YAML or JSON
  - Editors:
    - Swagger Editor http://editor.swagger.io 
    - Visual Studio Marketplace: https://marketplace.visualstudio.com/items?itemName=Arjun.swagger-viewer
- Can be easily shared
- Can generate code shells from the document
- Create the document after identifying the requirements
## 4.2 Describing API resources and actions with OAS
- Consists of a Header and paths
- Header
  ```
    openapi: "3.0.0"  
    info:  
	    title: Shopping API  
	    version: "1.0"
  ```
- Path
   ```
      paths:  
	      /products:  
		      get:  
			      summary: Search for products  
			      description: |Search using a free query (query parameter)  
			      parameters:  
			      requestbody:  
			      responses:  
		      post:  
			      summary:  
			      description:  
			      requestbody:  
			      responses:
    ```
## 4.3 Describing API data with OpenAPI and JSON Schema
- OAS relies on JSON Schema specification to describe all data
- Query
  ```
  parameters:  
	  - name: free-query  
	  description: | A product's name, reference, or partial description  
	  in: query  
	  required: false  
	  schema:  
		  type: string
  ```
- Post
  ```
  requestBody:  
	  description: Product's information   
	  content:  
		  application/json:  
	  schema:  
		  type: object  
		  description: A product  
		  required:  
			  - reference  
			  - name  
			  - price  
		  properties:  
			  reference:  
				  type: string  
				  description: Product's unique identifier  
				  example: ISBN-9781617295102  
			  name:  
				  type: string  
				  example: The Design of Web APIs  
			  price:  
				  type: number  
				  example: 44.99  
			  description:  
				  type: string  
				  example: A book about API design
  ```
## 4.4 Describing an API efficiently with OAS
- Components, such as schemas, parameters, and responses can be reusable
- Use the key *component*
- Refer to it using $ref and the path/url to the component
  - *$ref: "#/components/schemas/product"*
