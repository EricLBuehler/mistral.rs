from openai import OpenAI

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

c_lark = r"""
# start: program
# we generate only one
start: function_def

program: (function_def)*

function_def: type IDENTIFIER "(" params ")" "{" stmt* "}"
params: param ("," param)*
      | 
param: type IDENTIFIER
type: "int" | "float" | "char" | "void"

stmt: var_decl
    | expr_stmt
    | return_stmt
    | if_stmt
    | while_stmt
    | block

var_decl: type IDENTIFIER ("=" expr)? ";"
expr_stmt: expr ";"
return_stmt: "return" expr ";"
if_stmt: "if" "(" expr ")" stmt ("else" stmt)?
while_stmt: "while" "(" expr ")" stmt
block: "{" stmt* "}"

expr: expr ("+" | "-" | "*" | "/" | "==" | "!=" | "<" | ">" | "<=" | ">=") expr
    | "(" expr ")"
    | IDENTIFIER
    | NUMBER
    | STRING
    | IDENTIFIER "(" args ")"
args: expr ("," expr)*
    |

IDENTIFIER: /[a-zA-Z_][a-zA-Z_0-9]*/
NUMBER: /[0-9]+(\.[0-9]+)?/
STRING: /"[^"]*"/

%import common.WS
%ignore WS
"""

completion = client.chat.completions.create(
    model="default",
    messages=[
        {
            "role": "user",
            "content": "Write the main function in C, returning 42. Answer with just the code, no explanation.",
        }
    ],
    max_tokens=256,
    frequency_penalty=1.0,
    top_p=0.1,
    temperature=0,
    extra_body={"grammar": {"type": "lark", "value": c_lark}},
)

print(completion.choices[0].message.content)
