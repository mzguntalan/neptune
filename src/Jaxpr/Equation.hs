module Jaxpr.Equation where

-- responsible for making Equation
import Jaxpr.Grammar

compileLaxToEquation :: LaxPrimitive -> Equation
compileLaxToEquation (Abs v (Var o)) = Equation [o] "abs" [] [Var v]
compileLaxToEquation (Add x y (Var o)) = Equation [o] "add" [] [Var x, Var y]
compileLaxToEquation (Concatenate vars params (Var o)) = Equation [o] "concatenate" params arguments
  where
    arguments = map Var vars
compileLaxToEquation _ = error "Not implemented yet."
