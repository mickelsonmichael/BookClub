function getCapability(url, callback) {
    let capabilityUrl = new URL(url);
    let token = capabilityUrl.hash.substring(1); // remove the #

    capabilityUrl.hash = ''; // remove the hash so it isn't sent accidentally
    capabilityUrl.search = '?access_token=' + token; // move the token from the fragment to the query

    return fetch(capabilityUrl.href)
        .then(response => response.json())
        .then(callback)
        .catch(error => console.error("Error getting capability: ", error));
}