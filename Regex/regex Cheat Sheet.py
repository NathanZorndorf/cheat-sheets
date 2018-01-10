. => wild card
	# Example :
	...\. => match asd.

\d => any digit 

[] => match any character in brackets


[^] => match any except the character(s) in brackets
	# Example :
	[^b]og # match words like hog, dog, but NOT bog

z{3}   => matches the character 'z' 3 times
z{3,}  => matches the character 'z' 3 or more times 
z{3,6} => matches the character 'z' no less than 3 times and no more than 6
[xyz]{3} => matches x y or z 3 times


\d* => matches any number of digits 
\d+ => matches any number of digits, but at least 1
[abc]* => matches any number of 'a's, 'b's, or 'c's, as long as they are in order 

\w =>  alpha-numeric and '_' : [A-Za-z0-9_]

z? => denotes that the 'z' is optional, or in other words, fine 0 or one instance or character. 
	# Example:
	xyz? => 'xyz' and 'xy' would both match 


\b => boundary (end, beginning, or boundary, which occurs between a word and non-word character)
	# Example:
	\bfoo\b => will match 'foo', 'foo.', but not 'foobar'

	\s

\n 
\t
\r
\_ => matches space????
\s => matches any of the white space characters

^success => ONLY match a line that begins with 'success'
	# Example:
	^foo => 'foo','foobar' would match, but 'bazfoo' would not

$success => ONLY match a line that ends with 'success'
	# Example:
	$foo => 'foo' would match, but 'foobar' would not




