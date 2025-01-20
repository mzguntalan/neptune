{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE StandaloneDeriving #-}

module Jaxpr.Clax where

import Jaxpr.Equation
import Jaxpr.Grammar
import Jaxpr.Lax as Jlax

-- composable lax functions
-- this will be the comfortable low level
-- Jaxpr.Lax are constructors for LaxPrimitive
-- while Clax will mirror Lax, it will handle arbitrary composition/chaining of lax functions

-- a ClaxExpression will be the only type that clax functions can accept (and finally return)
-- all ClaxExpression should be compilable to a JaxExpr

data ClaxExpression = ClaxExpression [ConstVar] [InputVar] [LaxPrimitive] [Expression] -- maybe JaxExpr is the right level of composability

var :: Variable -> ClaxExpression
var v = ClaxExpression [] [] [] [Var v]

unwrapExpressionToVariable :: Expression -> Variable
unwrapExpressionToVariable (Var v) = v
unwrapExpressionToVariable (Lit l) = l

id :: ClaxExpression -> ClaxExpression
id (ClaxExpression _ _ _ [x]) = ClaxExpression [] [y] [] [Var y]
  where
    y = unwrapExpressionToVariable x
id _ = error "`id` can only accept one array"

add :: ClaxExpression -> ClaxExpression -> ClaxExpression
add (ClaxExpression _ _ _ [x]) (ClaxExpression _ _ _ [y]) = ClaxExpression [] [a, b] [prim] [Var o]
  where
    a = unwrapExpressionToVariable x
    b = unwrapExpressionToVariable y
    prim = Jlax.add a b
    Add _ _ exprO = prim
    o = unwrapExpressionToVariable exprO

compileClaxToJaxpr :: ClaxExpression -> JaxExpr
compileClaxToJaxpr (ClaxExpression constvars inputvars prims exprs) = JaxExpr constvars inputvars (map compileLaxToEquation prims) exprs

-- todo: at this moment, i realized Variable and Lit are just Arrays ... should probably refactor this soon
