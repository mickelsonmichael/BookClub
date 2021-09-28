# Growing APIs
- Designing APIs requires us to think beyond the APIs themselves because they are
just one part of the whole
## 13.1 The API lifecycle
- Analyze phase: *an idea has been presented, topics are explored, is it worth working on?*
- Design phase: *ideas from the analyze phase are investigated and a programming interface contract is created*
- Implementation phase: *application is built*
- The analyze, design and implentation process can be repeated
- Publish phase: *application is made available to the targeted consumers*
- Evolution: *new features are added*
- Retirement: *new version is created, API was not successful, API is no longer needed*
## 13.2 Building API design guidelines
- Design guidelines: set of rules that will be used by all designers
- Guidelines are needed to provide consistency throughout the API
- What to put in the guidelines
1. Reference: *focus on describing the foundations. The minimum API design guidelines that you must create: these list and describe all of the principles and rules*
2. Use Case: *how to apply the foundations using use cases. They provide ready to use "recipes" or solutions*
3. Design Process: *provide a design canvas or link to existing documentation or checklists*
4. Extended: *provide information about the surrounding processes like security, network, or implementation*
- When building the guidelines, start small. Get the basics in first and then add more information as you go
- Should be built by designers for designers
- http://webconcepts.info a place to learn about standards in one spot
- http://apistylebook.com collection of api styles for inspiration
## 13.3 Reviewing APIs
- Apis need to be reviewed throughout the lifecycle to make sure they are working as intended.
- Analyze the needs (checklist on page 345)
- Linting the design (checklist on page 347)
- Review the design from the provider's perspective (checklist on page 349)
- Review the design from the consumer's perspective (checklist on page 351)
- Verify the implementation (unit test) *be careful about testing using generated documentation*
- Involve the entire network chain in testing
## 13.4 Communication and sharing
- make your documentation easily accessable to those who need to use it
