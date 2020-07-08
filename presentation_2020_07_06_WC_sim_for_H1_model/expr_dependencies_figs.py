# generate graphviz diagrams for presentation
from wc_lang import (DfbaObjectiveExpression, ObservableExpression, FunctionExpression, StopConditionExpression, RateLawExpression)

print('edge [arrowhead = open];')
for expr_class in (DfbaObjectiveExpression, ObservableExpression, FunctionExpression, StopConditionExpression, RateLawExpression):
    expr_class_name_stripped = expr_class.__name__.replace('Expression', '')
    print(f'"{expr_class_name_stripped}" [style="filled,bold"];')
    for expression_term_model in expr_class.Meta.expression_term_models:
        print(f'"{expression_term_model}" -> "{expr_class_name_stripped}"')
print('"Reaction" -> "Species" [style=dashed]')
print('"Species" -> "Compartment" [style=dashed]')
print('"Reaction" [fillcolor=DarkOrange, style=filled];')
print('"RateLaw" [fillcolor=DeepSkyBlue, color="black", style="filled,bold"];')

print()
print('digraph G { // Expression terms')
print('edge [arrowhead = open];')
for expr_class in (ObservableExpression, FunctionExpression, StopConditionExpression, RateLawExpression):
    expr_class_name_stripped = expr_class.__name__.replace('Expression', '')
    print(f'"{expr_class_name_stripped}" [style="filled,bold"];')
    for expression_term_model in expr_class.Meta.expression_term_models:
        print(f'"{expr_class_name_stripped}" -> "{expression_term_model}"')
print('"Compartment" -> "Species" [style=dashed]')
print('}')
