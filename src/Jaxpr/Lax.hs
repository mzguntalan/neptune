{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE StandaloneDeriving #-}

module Jaxpr.Lax where

import Jaxpr.Interpreter

-- reference: jax.lax module
-- provides the primitives and the
--
--

applyLax :: LaxPrimitive -> [Parameter] -> [Variable] -> JaxExpr
applyLax Abs [] [var] = applyAsUnary Abs [] var
applyLax Abs [] vars = error ("Abs received the wrong number " ++ show (length vars) ++ " of params should be 1")
applyLax Abs params [] = error ("Abs received the wrong number " ++ show (length params) ++ " of params should be 0")
applyLax _ _ _ = error "Not Implemented"

applyAsUnary :: LaxPrimitive -> [Parameter] -> Variable -> JaxExpr
applyAsUnary unary params var = JaxExpr [] [var] [eq] [Var out]
  where
    out = Variable "out" vartype
    (Variable _ vartype) = var
    eq = Equation [out] unary params [Var var]

-- applyLax _ _ _ = error "Not Implemented"
