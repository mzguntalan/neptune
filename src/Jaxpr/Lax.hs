{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE StandaloneDeriving #-}

module Jaxpr.Lax where

import Data.Version (Version (versionTags))
import Jaxpr.Grammar

-- this aims to expose the same functions as in jax.lax
-- validation should go here

abs :: Variable -> LaxPrimitive
abs (Variable varname vartype) = case vartype of
    (F32 _) -> Abs var (Var out)
    (Str _) -> error "`abs` not applicable to Str"
    _ -> error "Not implemented"
  where
    out = Variable "out" vartype
    var = Variable varname vartype

add :: Variable -> Variable -> LaxPrimitive
add (Variable varname1 vartype1) (Variable varname2 vartype2)
    | vartype1 == vartype2 = Add x y (Var out) -- string hasn't been handled
    | otherwise = error "`vartype`s should be equal"
  where
    x = Variable varname1 vartype1
    y = Variable varname2 vartype2
    out = Variable "out" vartype1

concatenate :: [Variable] -> Int -> LaxPrimitive
concatenate [] _ = error "`concatenate` cannot operate on an empty list"
concatenate (v : vs) axis = Concatenate vars [Parameter "dimension" (show axis)] (Var out)
  where
    vars = v : vs
    Variable _ vartype = v
    out = Variable "out" vartype -- WRONG VARTYPE!
