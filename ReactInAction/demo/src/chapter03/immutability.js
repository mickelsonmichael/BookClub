// This example demonstrates some of the ways you can modify an array
//
// The first set are mutable changes; the changes are made "in place" on the original array
// The second set are immutable changes; the original array is "intact" (although not the way I did it)
//
// The direct modifications are a bit contrived, since it both cases `numbers` has been modified
// More important are the sorting functions:
//
// The first will result in the original reference being sorted, since both `n` and `numbers` reference
// the same spot in memory. This means the function has "side effects". You didn't expect it to sort
// the original array, only to return a sorted version
//(this is less obvious a problem in pure JS since there is no return type for functions until runtime)
//
// The second sort function will leave `numbers` in the original state it was passed in
// It creates a copy and returns the sorted version of that new copy
// The user of the function is then able to decide if they want to overwrite their array
// or keep it intact and use the sorted array in another way

let numbers = [5,4,3,2,1];

// using mutability

numbers[2] = 3; // update array
numbers.pop(); // remove from the end
numbers.shift(); // remove from the beginning
numbers.pop(6); // add to the end
numbers.unshift(7); // add to the beginning
numbers.splice(4,1); // remove one from the array

function sortNumbers(n) {
    n.sort();
    return n; // the reference to numbers is now modified
}

sortNumbers(numbers);

numbers = [5,4,3,2,1]; // reset

// using immutability

numbers = [...numbers.slice(0,1), 3, ...numbers.slice(3)]; // update
numbers = [...numbers.slice(0, numbers.length - 1)]; // pop
numbers = [...numbers.slice(1)]; // shift
numbers = [...numbers, 6] // pop
numbers = [7, ...numbers] // unshift
numbers = [...numbers.slice(0, 3), ...numbers.slice(5)]; // splice

function sortNumbersImmutably(n) {
    let newN = [...n];
    newN.sort();
    return newN; // the original numbers is still intact
}

const sorted = sortNumbersImmutably(numbers);
