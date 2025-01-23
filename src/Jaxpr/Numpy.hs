module Jaxpr.Numpy where

import Data.Maybe (isNothing)
import Jaxpr.Blx (BlxTrace)
import Prelude hiding (abs, all, any, max, min)

-- This module has automatic convertions to BlxPrimitives

type NumpyArray = BlxTrace

-- ndarray.at
abs :: NumpyArray -> NumpyArray
abs = absolute

absolute :: NumpyArray -> NumpyArray
absolute x = _

acos :: NumpyArray -> NumpyArray
acos x = _

acosh :: NumpyArray -> NumpyArray
acosh x = _

add :: NumpyArray -> NumpyArray -> NumpyArray
add x y = _

all :: NumpyArray -> Maybe Int -> Maybe Bool -> Maybe NumpyArray -> NumpyArray
all a axis keepdims whereArray
    | isNothing keepdims = all a axis (Just False) whereArray
    | otherwise = _

allClose :: NumpyArray -> NumpyArray -> Maybe Float -> Maybe Float -> Maybe Bool -> NumpyArray
allClose a b rtol atol equalNan
    | isNothing rtol = allClose a b (Just 0.000_01) atol equalNan
    | isNothing atol = allClose a b rtol (Just 0.000_000_01) equalNan
    | isNothing equalNan = allClose a b rtol atol (Just False)
    | otherwise = _

amax :: NumpyArray -> Maybe Int -> Maybe Bool -> Maybe NumpyArray -> Maybe NumpyArray
amax = max

amin :: NumpyArray -> Maybe Int -> Maybe Bool -> Maybe NumpyArray -> Maybe NumpyArray
amin = min

angle :: NumpyArray -> MaybeBool -> NumpyArray
angle z deg
    | isNothing deg = angle z (Just False)
    | otherwise = _

any :: NumpyArray -> Maybe Int -> Maybe Bool -> Maybe NumpyArray -> Maybe NumpyArray
any a axis keepdims whereArray
    | isNothing keepdims = any a axis (Just False) whereArray
    | otherwise = _

append :: NumpyArray -> NumpyArray -> Maybe Int -> NumpyArray
append arr values axis = _

max :: NumpyArray -> Maybe Int -> Maybe Bool -> Maybe NumpyArray -> Maybe NumpyArray -> Maybe NumpyArray
max a axis keepdims initial whereArray
    | isNothing keepdims = max a axis (Just False) initial whereArray
    | otherwise = _

min :: NumpyArray -> Maybe Int -> Maybe Bool -> Maybe NumpyArray -> Maybe NumpyArray -> Maybe NumpyArray
min a axis keepdims initial whereArray
    | isNothing keepdims = min a axis (Just False) initial whereArray
    | otherwise = _
