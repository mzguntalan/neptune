{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE StandaloneDeriving #-}

module Jaxpr.Interpreter where

-- related to compiling down to JAXPR goes here
-- TODO: avoid using show ; show is for haskell (development)
-- use another function to compile to Jaxpr components

import Data.List (intercalate)
import Jaxpr.Grammar

compileEquationToJaxprEquationComponent :: Equation -> String
compileEquationToJaxprEquationComponent (Equation vars prim params exprs) = unwords components
  where
    components = [varsShow, "=", parameterizedPrim, exprsShow]
    varsShow = unwords (map show vars)
    parameterizedPrim = prim ++ paramsShow
    paramsShow = case params of
        [] -> ""
        _ -> "[" ++ unwords (map show params) ++ "]"
    exprsShow = unwords (map show exprs)

instance Show Equation where
    show = compileEquationToJaxprEquationComponent

compileSingleEquationToJaxpr :: Equation -> JaxExpr
compileSingleEquationToJaxpr (Equation outs prim params inputs) = JaxExpr [] (map f inputs) [eq] (map Var outs)
  where
    eq = Equation outs prim params inputs
    f (Var v) = v
    f (Lit _) = error "I don't think there are literals here"

compile :: JaxExpr -> String
compile (JaxExpr constvars inputvars equations expressions) = "{ lambda " ++ constvarsShow ++ " ; " ++ inputvarsShow ++ ". let\n" ++ equationsShow ++ "\n in " ++ expressionsShow ++ " }"
  where
    constvarsShow = unwords (map show constvars)
    inputvarsShow = unwords (map show inputvars)
    equationsShow = intercalate "\n" (map (("\t" ++) . show) equations)
    expressionsShow = "(" ++ intercalate "," (map show expressions) ++ ",)"

instance Show JaxExpr where
    show = compile
