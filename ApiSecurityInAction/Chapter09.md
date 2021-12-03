# Capability-based Security and Macaroons

- (Michael's Notes)[#michael]

## Michael

While identity-based access control is great, it doesn't really mesh well with sharing. If a user wants to show a message in Natter to another user, they'd either have to grant that user permissions, which is cumbersome and exposes potential vulnerabilities, or find some workaround to get their message shared.

Capability-based security seeks to solve that problem, and to rectify ["confused deputy"](https://en.wikipedia.org/wiki/Confused_deputy_problem) attacks. A CSRF attack is an example of a confused deputy attack, since the attack is utilizing a patsy with greater access to perform some action they'd normally not be allowed to do.
