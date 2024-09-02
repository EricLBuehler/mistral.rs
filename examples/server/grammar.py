from openai import OpenAI

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

JSON_KBNF = '''
(* JSON Grammar *)

(* JSON text must contain a single JSON value *)
start       = value ;

(* A JSON value can be an object, array, string, number, true, false, or null *)
value      ::= object 
           | array 
           | string 
           | number 
           | "true" 
           | "false" 
           | "null" ;

(* A JSON object is a collection of key/value pairs enclosed in curly braces *)
object     ::= "{" [ members ] "}" ;
members    ::= pair { "," pair } ;
pair       ::= string ":" value ;

(* A JSON array is an ordered list of values enclosed in square brackets *)
array      ::= "[" [ elements ] "]" ;
elements   ::= value { "," value } ;

(* A JSON string is a sequence of Unicode characters enclosed in double quotes *)
string     ::= "\"" { character } "\"" ;
character  ::= escape 
           | non_escape ;

(* Escape sequences *)
escape     ::= "\\" ( "\"" | "\\" | "/" | "b" | "f" | "n" | "r" | "t" | "u" hex hex hex hex ) ;
non_escape ::= ? any character except " or \ or control characters ? ;

(* A JSON number is an integer or floating-point number *)
number     ::= integer [ fraction ] [ exponent ] ;
integer    ::= digit | "-" digit | "-" non_zero_digit { digit } | non_zero_digit { digit } ;
fraction   ::= "." { digit } ;
exponent   ::= ("e" | "E") [ "+" | "-" ] { digit } ;
digit      ::= "0" | non_zero_digit ;
non_zero_digit ::= "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" ;

(* Hexadecimal digits for unicode escape sequences *)
hex        ::= digit | "A" | "B" | "C" | "D" | "E" | "F" ;

'''

EXPR_KBNF = '''

(* Grammar for Mathematical Expressions *)

(* An expression can be a term or a sum/subtraction of terms *)
expression   ::= term { ("+" | "-") term } ;

(* A term can be a factor or a product/division of factors *)
term         ::= factor { ("*" | "/") factor } ;

(* A factor can be a number, a variable, or a parenthesized expression *)
factor       ::= number 
             | variable 
             | "(" expression ")" ;

(* A number is a sequence of digits, possibly with a decimal point *)
number       ::= digit { digit } [ "." digit { digit } ] ;

(* A variable is an identifier starting with a letter, possibly followed by letters or digits *)
variable     ::= letter { letter | digit } ;

(* Digits and letters *)
digit        ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" ;
letter       ::= "a" | "b" | "c" | "d" | "e" | "f" | "g" | "h" | "i" | "j"
             | "k" | "l" | "m" | "n" | "o" | "p" | "q" | "r" | "s" | "t"
             | "u" | "v" | "w" | "x" | "y" | "z"
             | "A" | "B" | "C" | "D" | "E" | "F" | "G" | "H" | "I" | "J"
             | "K" | "L" | "M" | "N" | "O" | "P" | "Q" | "R" | "S" | "T"
             | "U" | "V" | "W" | "X" | "Y" | "Z" ;


'''

completion = client.chat.completions.create(
    model="mistral",
    messages=[
        {
            "role": "user",
            "content": "Write a mathematical expression.",
        }
    ],
    max_tokens=256,
    frequency_penalty=1.0,
    top_p=0.1,
    temperature=0,
    extra_body={"grammar": {"type": "kbnf", "value": EXPR_KBNF}},
)

print(completion.choices[0].message.content)
