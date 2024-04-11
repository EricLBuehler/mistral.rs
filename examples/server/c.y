// based on http://www.lysator.liu.se/c/ANSI-C-grammar-y.html

%start translation_unit
%%

SKIP
    : "//\*[^*]*\*+([^/*][^*]*\*+)*//"  // block comment
    | "///.*/"                          // line comment
    | "/\n[ \t\v\f]*#(.*\\\n)*.*/"      // pre-processor
    | "/\n?[ \t\v\f]*/"                 // white-space
    ;

IDENTIFIER: "/[a-zA-Z_][0-9a-zA-Z_]*/" ;

TYPE_NAME: "/[a-zA-Z_][0-9a-zA-Z_]*_t/" ;

CONSTANT
        : "/0[xX][0-9a-fA-F]+[uUlL]*?/"
        | "/0[0-9]+[uUlL]*?/"
        | "/[0-9]+[uUlL]*?/"
        | "/[a-zA-Z_]?'(\\.|[^\\'])+'/"
        | "/[0-9]+[Ee][+-]?[0-9]+[flFL]?/"
        | "/[0-9]*\\.[0-9]+([Ee][+-]?[0-9]+)?[flFL]?/"
        | "/[0-9]+\\.[0-9]*([Ee][+-]?[0-9]+)?[flFL]?/"
        ;

STRING_LITERAL: '/[a-zA-Z_]?"(\\.|[^\\"])*"/' ;

primary_expression
    : IDENTIFIER
    | CONSTANT
    | STRING_LITERAL
    | "(" expression ")"
    ;

postfix_expression
    : primary_expression
    | postfix_expression "[" expression "]"
    | postfix_expression "(" ")"
    | postfix_expression "(" argument_expression_list ")"
    | postfix_expression "." IDENTIFIER
    | postfix_expression "->" IDENTIFIER
    | postfix_expression "++"
    | postfix_expression "--"
    ;

argument_expression_list
    : assignment_expression
    | argument_expression_list "," assignment_expression
    ;

unary_expression
    : postfix_expression
    | "++" unary_expression
    | "--" unary_expression
    | unary_operator cast_expression
    | "sizeof" unary_expression
    | "sizeof" "(" type_name ")"
    ;

unary_operator
    : "&"
    | "*"
    | "+"
    | "-"
    | "~"
    | "!"
    ;

cast_expression
    : unary_expression
    | "(" type_name ")" cast_expression
    ;

multiplicative_expression
    : cast_expression
    | multiplicative_expression "*" cast_expression
    | multiplicative_expression "/" cast_expression
    | multiplicative_expression "%" cast_expression
    ;

additive_expression
    : multiplicative_expression
    | additive_expression "+" multiplicative_expression
    | additive_expression "-" multiplicative_expression
    ;

shift_expression
    : additive_expression
    | shift_expression "<<" additive_expression
    | shift_expression ">>" additive_expression
    ;

relational_expression
    : shift_expression
    | relational_expression "<" shift_expression
    | relational_expression ">" shift_expression
    | relational_expression "<=" shift_expression
    | relational_expression ">=" shift_expression
    ;

equality_expression
    : relational_expression
    | equality_expression "==" relational_expression
    | equality_expression "!=" relational_expression
    ;

and_expression
    : equality_expression
    | and_expression "&" equality_expression
    ;

exclusive_or_expression
    : and_expression
    | exclusive_or_expression "^" and_expression
    ;

inclusive_or_expression
    : exclusive_or_expression
    | inclusive_or_expression "|" exclusive_or_expression
    ;

logical_and_expression
    : inclusive_or_expression
    | logical_and_expression "&&" inclusive_or_expression
    ;

logical_or_expression
    : logical_and_expression
    | logical_or_expression "||" logical_and_expression
    ;

conditional_expression
    : logical_or_expression
    | logical_or_expression "?" expression ":" conditional_expression
    ;

assignment_expression
    : conditional_expression
    | unary_expression assignment_operator assignment_expression
    ;

assignment_operator
    : "="
    | "*="
    | "/="
    | "%="
    | "+="
    | "-="
    | "<<="
    | ">>="
    | "&="
    | "^="
    | "|="
    ;

expression
    : assignment_expression
    | expression "," assignment_expression
    ;

constant_expression
    : conditional_expression
    ;

declaration
    : declaration_specifiers ";"
    | declaration_specifiers init_declarator_list ";"
    ;

declaration_specifiers
    : storage_class_specifier
    | storage_class_specifier declaration_specifiers
    | type_specifier
    | type_specifier declaration_specifiers
    | type_qualifier
    | type_qualifier declaration_specifiers
    ;

init_declarator_list
    : init_declarator
    | init_declarator_list "," init_declarator
    ;

init_declarator
    : declarator
    | declarator "=" initializer
    ;

storage_class_specifier
    : "typedef"
    | "extern"
    | "static"
    | "auto"
    | "register"
    | "inline"
    ;

type_specifier
    : "void"
    | "char"
    | "short"
    | "int"
    | "long"
    | "float"
    | "double"
    | "signed"
    | "unsigned"
    | "bool"
    | struct_or_union_specifier
    | enum_specifier
    | TYPE_NAME
    ;

struct_or_union_specifier
    : struct_or_union IDENTIFIER "{" struct_declaration_list "}"
    | struct_or_union "{" struct_declaration_list "}"
    | struct_or_union IDENTIFIER
    ;

struct_or_union
    : "struct"
    | "union"
    ;

struct_declaration_list
    : struct_declaration
    | struct_declaration_list struct_declaration
    ;

struct_declaration
    : specifier_qualifier_list struct_declarator_list ";"
    ;

specifier_qualifier_list
    : type_specifier specifier_qualifier_list
    | type_specifier
    | type_qualifier specifier_qualifier_list
    | type_qualifier
    ;

struct_declarator_list
    : struct_declarator
    | struct_declarator_list "," struct_declarator
    ;

struct_declarator
    : declarator
    | ":" constant_expression
    | declarator ":" constant_expression
    ;

enum_specifier
    : "enum" "{" enumerator_list "}"
    | "enum" IDENTIFIER "{" enumerator_list "}"
    | "enum" IDENTIFIER
    ;

enumerator_list
    : enumerator
    | enumerator_list "," enumerator
    ;

enumerator
    : IDENTIFIER
    | IDENTIFIER "=" constant_expression
    ;

type_qualifier
    : "const"
    | "volatile"
    ;

declarator
    : pointer direct_declarator
    | direct_declarator
    ;

direct_declarator
    : IDENTIFIER
    | "(" declarator ")"
    | direct_declarator "[" constant_expression "]"
    | direct_declarator "[" "]"
    | direct_declarator "(" parameter_type_list ")"
    | direct_declarator "(" identifier_list ")"
    | direct_declarator "(" ")"
    ;

pointer
    : "*"
    | "*" type_qualifier_list
    | "*" pointer
    | "*" type_qualifier_list pointer
    ;

type_qualifier_list
    : type_qualifier
    | type_qualifier_list type_qualifier
    ;


parameter_type_list
    : parameter_list
    | parameter_list "," "..."
    ;

parameter_list
    : parameter_declaration
    | parameter_list "," parameter_declaration
    ;

parameter_declaration
    : declaration_specifiers declarator
    | declaration_specifiers abstract_declarator
    | declaration_specifiers
    ;

identifier_list
    : IDENTIFIER
    | identifier_list "," IDENTIFIER
    ;

type_name
    : specifier_qualifier_list
    | specifier_qualifier_list abstract_declarator
    ;

abstract_declarator
    : pointer
    | direct_abstract_declarator
    | pointer direct_abstract_declarator
    ;

direct_abstract_declarator
    : "(" abstract_declarator ")"
    | "[" "]"
    | "[" constant_expression "]"
    | direct_abstract_declarator "[" "]"
    | direct_abstract_declarator "[" constant_expression "]"
    | "(" ")"
    | "(" parameter_type_list ")"
    | direct_abstract_declarator "(" ")"
    | direct_abstract_declarator "(" parameter_type_list ")"
    ;

initializer
    : assignment_expression
    | "." IDENTIFIER "=" assignment_expression
    | "[" assignment_expression "]" "=" assignment_expression
    | "{" initializer_list "}"
    | "{" initializer_list "," "}"
    ;

initializer_list
    : initializer
    | initializer_list "," initializer
    ;

statement
    : labeled_statement
    | compound_statement
    | expression_statement
    | selection_statement
    | iteration_statement
    | jump_statement
    ;

labeled_statement
    : IDENTIFIER ":" statement
    | "case" constant_expression ":" statement
    | "default" ":" statement
    ;

compound_statement
    : "{" "}"
    | "{" statement_list "}"
    ;

declaration_list
    : declaration
    | declaration_list declaration
    ;

statement_or_declaration
    : statement
    | declaration
    ;

statement_list
    : statement_or_declaration
    | statement_list statement_or_declaration
    ;

expression_statement
    : ";"
    | expression ";"
    ;

for_decl
    : expression_statement
    | declaration
    ;

selection_statement
    : "if" "(" expression ")" statement
    | "if" "(" expression ")" statement "else" statement
    | "switch" "(" expression ")" statement
    ;

iteration_statement
    : "while" "(" expression ")" statement
    | "do" statement "while" "(" expression ")" ";"
    | "for" "(" for_decl expression_statement ")" statement
    | "for" "(" for_decl expression_statement expression ")" statement
    ;

jump_statement
    : "goto" IDENTIFIER ";"
    | "continue" ";"
    | "break" ";"
    | "return" ";"
    | "return" expression ";"
    ;

translation_unit
    : external_declaration
    | translation_unit external_declaration
    ;

external_declaration
    : function_definition
    | declaration
    ;

function_definition
    : declaration_specifiers declarator declaration_list compound_statement
    | declaration_specifiers declarator compound_statement
    | declarator declaration_list compound_statement
    | declarator compound_statement
    ;

%%
