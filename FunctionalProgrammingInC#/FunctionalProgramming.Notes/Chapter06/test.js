
var myStuff = [ "hello", "goodbye" ];

myStuff.forEach(console.log)
    .map(m => m.toUpperCase())
    .forEach(console.log)
    .filter(m => m.length > 3);

for (let a of myStuff)
{

}
