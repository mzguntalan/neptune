{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE StandaloneDeriving #-}

module Jaxpr.Clax where

import Jaxpr.Grammar

-- composable lax functions
-- this will be the comfortable low level
-- Jaxpr.Lax are constructors for LaxPrimitive
-- while Clax will mirror Lax, it will handle arbitrary composition/chaining of lax functions

-- a ClaxExpression will be the only type that clax functions can accept (and finally return)
-- all ClaxExpression should be compilable to a JaxExpr

data ClaxExpression = ClaxExpression [ConstVar] [InputVar] [Equation] [Expression] -- maybe JaxExpr is the right level of composability

id :: ClaxExpression -> ClaxExpression
id (ClaxExpression _ _ _ [x]) = ClaxExpression [] [y] [] [Var y]
  where
    f :: Expression -> Variable
    f (Var v) = Variable v
    f (Lit l) = Variable l
    y = f x
