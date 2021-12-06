# Capability-based Security and Macaroons

- (Michael's Notes)[#michael]

## Michael

While identity-based access control is great, it doesn't really mesh well with sharing. If a user wants to show a message in Natter to another user, they'd either have to grant that user permissions, which is cumbersome and exposes potential vulnerabilities, or find some workaround to get their message shared.

Capability-based security seeks to solve that problem, and to rectify ["confused deputy"](https://en.wikipedia.org/wiki/Confused_deputy_problem) attacks. A CSRF attack is an example of a confused deputy attack, since the attack is utilizing a patsy with greater access to perform some action they'd normally not be allowed to do.

### Capability-based security (Section 9.1)

The exact definition of capability-based security can be a little vague and examples given are often in terms of OS security, rather than APIs (although Madden gives an example in section 9.2). But at its core, a capability is "an unforgeable reference to an object or resource together with a set of permissions to access that resource" (Madden, 2020). Or, more simply, it is an identifier for an object, such as an ID, and a set of permissions a user with this capability would have access to.

Madden provides an example of a UNIX filesystem, which I won't repeat here, and then lists a couple advantages and disadvantages to capabilities verses identity-based access control.

1. Capabilities can not be forged, and you cannot send a request for a resource you can't access. Meanwhile in an identity-based approach, anyone can request the action and be either approved or denied
2. Capabilities are more fine-grained, and permissions can be granted in smaller, more *principle of least privilege*-friendly way
3. It is harder to list who has access to what resources with a capability-based system. You can determine that someone shared a capability, but you cannot easily determine who they shared with with
4. Revoking access using a capability-based system is more complicated and not always possible. It also may be too broad in scope and revocations may result in too many capabilities being removed.

Check out the article [Capability Myths Demolished](https://srl.cs.jhu.edu/pubs/SRL2003-02.pdf) (M.S. Miller, K. Yee, J. Shapiro, 2003) for some assurances that capability-based systems are actually secure.