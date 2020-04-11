# Chapter 9 - Stringy Features

## Intro
- it’s worth remembering that the strings themselves haven’t changed at all. Both features provide new ways of obtaining strings, but that’s all. 
  - pg 252, para.1

## 9.1
- An *alignment*, which specifies a minimum width 
  - pg 254, para. 2
- A comma in the format item indicates an alignment, and a colon indicates a format string 
  - pg 254, para.4\
- …*ocalization* is the task of making sure your code does the right thing for all your users, no matter where they are in the world 
  - pg 255, para.3
- In .NET, the most important type to know about for localization purposes is Culture-Info. 
  - pg 255, para.4
- Often, you won’t see CultureInfo in a method signature, but instead the IFormatProvider interface, which CultureInfo implements 
  - pg 255, para. 5
- If you don’t specify a format provider, or if you pass null as the argument corresponding to an IFormatProvider parameter, CultureInfo.CurrentCulture will be used as a default. 
  - pg 257, para.2
- For machine-to-machine communication… you should use the invariant culture 
  - pg 257 para. 4
  
## 9.2
- with interpolated string literals, you specify the values and their formatting information inline 
  - pg 258, para.2
- If the value implements the IFormattable interface, its ToString(string, IFormatProvider) method will be called; otherwise, System .Object.ToString() is used. 
  - pg 259, para. 4
- Verbatim string literals are typically used for the following: Strings breaking over multiple lines, Regular expressions (which use backslashes for escaping, quite separate from the escaping the C# compiler uses in regular string literals), Hardcoded Windows filenames    - pg 260, para. 1

## 9.3
- [FormattableString] a class in the System namespace introduced in .NET 4.6 (and .NET Standard 1.3 in the .NET Core world). It holds the composite format string and the values so they can be formatted in whatever culture you want later. 
  - pg 262, para. 2
- there are conversions from interpolated string literal expressions to both FormattableString and IFormattable 
  - pg 262, para. 4
- But instead of string.Format, it calls the static Create method on the System.Runtime.CompilerServices.FormattableStringFactory class.    - pg 263, para. 1
- FormattableString is an abstract class… 
  - pg 263, para. 3
- You want to use the invariant culture to avoid any unexpected results from using the default culture 
  - pg 264, para. 2
- If you want to format a FormattableString in any culture other than the invariant one, you need to use one of the ToString methods. 
  - pg 265, para. 2
- …so any method accepting an IFormattable will accept a FormattableString 
  - pg 265, para. 3

## 9.4
- 
