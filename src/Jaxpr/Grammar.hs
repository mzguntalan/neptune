module Jaxpr.Grammar where

-- types and constructors for Grammar related
type VarName = String

type Shape = [Int]

data VarType = I32 Shape | I64 Shape | F32 Shape | F64 Shape | Str Shape deriving (Eq)

instance Show VarType where
    show (I32 s) = "i32" ++ show s
    show (I64 s) = "i64" ++ show s
    show (F32 s) = "f32" ++ show s
    show (F64 s) = "f64" ++ show s
    show (Str s) = "str" ++ show s

data Variable = Variable VarName VarType

instance Show Variable where
    show (Variable varname vartype) = varname ++ ":" ++ show vartype

type ParamName = String

type ParamValue = String

data Parameter = Parameter ParamName ParamValue

instance Show Parameter where
    show (Parameter name val) = name ++ "=" ++ val

type Literal = Variable

data Expression = Var Variable | Lit Literal

instance Show Expression where
    show (Var (Variable varname _)) = varname
    show (Lit lit) = show lit

-- if something is an array then it is a Variable
-- if something is a python primitive then it is a parameter
-- rule of thumb, actual verification to check behavior of jaxpr is needed per function i think
data LaxPrimitive
    = Abs Variable Expression
    | Acos Variable Expression
    | Acosh Variable Expression
    | Add Variable Variable Expression
    | Concatenate [Variable] [Parameter] Expression -- and more from jax.lax module

type LaxPrimitiveName = String

data Equation = Equation [Variable] LaxPrimitiveName [Parameter] [Expression] -- derives Show in Interpreter.hs

type ConstVar = Variable

type InputVar = Variable

data JaxExpr = JaxExpr [ConstVar] [InputVar] [Equation] [Expression] -- derives Show in Interpreter.hs
