module Jaxpr.Lax where

import Jaxpr.Interpreter

-- reference: jax.lax module
-- provides the primitives and the
--
--

labs :: Variable -> JaxExpr
labs (Variable varname vartype) = JaxExpr [] [x] [eq] [Var out]
  where
    out = Variable "out" vartype
    x = Variable varname vartype
    eq = Equation [out] Abs [] [Var x]

class (Primitive a) => UnaryPrimitiveNoParameter a

applyUnary :: (UnaryPrimitiveNoParameter a) => a -> Variable -> JaxExpr
applyUnary prim (Variable varname vartype) = JaxExpr [] [x] [eq] [Var out]
  where
    out = Variable "out" vartype
    x = Variable varname vartype
    eq = Equation [out] (toLaxPrimitive prim) [] [Var x]
