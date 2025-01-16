{-# LANGUAGE GADTs #-}

module Jaxpr where

import Data.List (intercalate)

type VarName = String

type Shape = [Integer]

data VarType = I32 Shape | I64 Shape | F32 Shape | F64 Shape

instance Show VarType where
    show (I32 s) = "i32" ++ show s
    show (I64 s) = "i64" ++ show s
    show (F32 s) = "f32" ++ show s
    show (F64 s) = "f64" ++ show s

data Variable = Variable VarName VarType

instance Show Variable where
    show (Variable varname vartype) = varname ++ ":" ++ show vartype

type ParamName = String

type ParamValue = String

data Parameter = Parameter ParamName ParamValue

instance Show Parameter where
    show (Parameter name val) = name ++ "=" ++ val

data Primitive = PrimAdd | PrimSub | PrimSin | PrimMul

instance Show Primitive where
    show PrimAdd = "add"
    show PrimSub = "sub"
    show PrimSin = "sin"
    show PrimMul = "mul"

type Literal = String

data Expression = Var Variable | Lit Literal

instance Show Expression where
    show (Var (Variable varname _)) = varname
    show (Lit lit) = lit

data Equation = Equation [Variable] Primitive [Parameter] [Expression]

instance Show Equation where
    show (Equation vars prim params exprs) = unwords components
      where
        components = ["let", varsShow, "=", parameterizedPrim, exprsShow]
        varsShow = unwords (map show vars)
        parameterizedPrim = show prim ++ paramsShow
        paramsShow = case params of
            [] -> ""
            _ -> "[" ++ unwords (map show params) ++ "]"
        exprsShow = unwords (map show exprs)

type ConstVar = Variable

type InputVar = Variable

data JaxExpr = JaxExpr [ConstVar] [InputVar] [Equation] [Expression]

instance Show JaxExpr where
    show (JaxExpr constvars inputvars equations expressions) = "{ lambda " ++ constvarsShow ++ " ; " ++ inputvarsShow ++ ". let\n" ++ equationsShow ++ "\n in " ++ expressionsShow ++ " }"
      where
        constvarsShow = unwords (map show constvars)
        inputvarsShow = unwords (map show inputvars)
        equationsShow = intercalate "\n" (map show equations)
        expressionsShow = "(" ++ intercalate "," (map show expressions) ++ ",)"
