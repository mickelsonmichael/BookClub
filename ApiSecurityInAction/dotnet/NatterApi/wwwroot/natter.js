const apiUrl = 'https://localhost:4567';

function createSpace(name, owner) {
    let data = {name: name, owner: owner};
    let token = localStorage.getItem('token');

    fetch(apiUrl + '/spaces', {
        method: 'POST',
        body: JSON.stringify(data),
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
        }
    })
    .then(response => {
        if (response.ok) {
            return response.json();
        } else if (response.status === 401) {
            window.location.replace('/login.html');
        } else {
            throw Error(response.statusText);
        }
    })
    .then(json => console.log('Created space: ', json.name, json.uri))
    .catch(error => console.error('Error: ', error));
}

function getCookie(cookieName) {
    const cookie = document.cookie.split(';')
        .map(item => {
            let splitterIndex = item.indexOf('=');

            let result = {
                key: item.substring(0, splitterIndex).trim(),
                value: decodeURIComponent(item.substring(splitterIndex + 1).trim())
            };

            return result;
        })
        .filter(item => item.key === cookieName)[0];

    if (cookie) {
        return cookie.value;
    }
}

window.addEventListener('load', function(e) {
    document.getElementById('createSpace')
        .addEventListener('submit', processFormSubmit);
});

function processFormSubmit(e) {
    e.preventDefault();

    let spaceName = document.getElementById('spaceName').value;
    let owner = document.getElementById('owner').value;

    createSpace(spaceName, owner);

    return false;
}
