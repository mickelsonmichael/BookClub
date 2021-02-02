const myDate = new Date();

// Method 1
const full: (s: string, n: number) => Date = (str, num) => {
    return myDate;
}

// Method 2
const semi = (str: string, num: number): Date => {
    return myDate;
}

// Method 3
const compact = (str: string, num: number) => {
    return myDate;
}

export { full, semi, compact };