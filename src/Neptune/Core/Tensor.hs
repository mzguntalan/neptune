{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Neptune.Core.Tensor where

import Data.List (intercalate)

data Abstract m c = (Eq m, Eq c) => Abstract m [c]

getM :: Abstract m c -> m
getM (Abstract m _) = m

getCs :: Abstract m c -> [c]
getCs (Abstract _ cs) = cs

abstractAreEqual :: Abstract m c -> Abstract m c -> Bool
abstractAreEqual (Abstract m1 cs1) (Abstract m2 cs2) = m1 == m2 && cs1 == cs2

class Operation a b where
    applyOperation :: a -> [b] -> [b]
    symbol :: a -> [b] -> String -- the symbol might be affected by the inputs::[b]; in all cases i know, it won't

data Computation a = forall p. (Eq a, Operation p a) => Computation p [a] [a]

instance Eq (Computation a) where
    (Computation p1 ins1 outs1) == (Computation p2 ins2 outs2) =
        symbol p1 ins1 == symbol p2 ins2
            && ins1 == ins2
            && outs1 == outs2

computation :: (Eq a, Operation p a) => p -> [a] -> Computation a
computation operation inputs = Computation operation inputs (applyOperation operation inputs)

applyOperationOnAbstract ::
    (Eq m, Operation p m) => p -> [Abstract m (Computation m)] -> [Abstract m (Computation m)]
applyOperationOnAbstract p abstractinputs = map wrap outputsMs
  where
    wrap o = Abstract o (newComputationByP : combinedComputations)
    newComputationByP = computation p inputs
    (Computation _ _ outputsMs) = newComputationByP
    inputs = map getM abstractinputs

    combinedComputations = concatMap getCs abstractinputs

data LaxPrimitiveType = Tf32 | Ti32 | Tstr deriving (Eq, Show)

data LaxTensorProperties = LaxTensorProperties [Int] LaxPrimitiveType deriving (Eq)

instance Show LaxTensorProperties where
    show (LaxTensorProperties s t) = show s ++ ":" ++ show t

type AbstractLaxTensor = Abstract LaxTensorProperties (Computation LaxTensorProperties)

instance Show AbstractLaxTensor where
    show (Abstract t cs) = "AbstractLaxTensor {" ++ show t ++ "}" ++ "[\n\t" ++ intercalate "\n\t" (map show cs) ++ "]"

mkAbstractLaxTensor :: [Int] -> LaxPrimitiveType -> AbstractLaxTensor
mkAbstractLaxTensor shape tensorType = Abstract i [computation Make [i]]
  where
    i = LaxTensorProperties shape tensorType

data Make = Make

instance Operation Make LaxTensorProperties where
    applyOperation Make [x] = [x]
    applyOperation _ _ = error "WrongNumberInputsError"
    symbol Make _ = "Make"

instance Show (Computation LaxTensorProperties) where
    show (Computation p inputs outputs) = os ++ " = " ++ symbol p inputs ++ " " ++ is
      where
        os = "(" ++ intercalate "," (map show inputs) ++ ")"
        is = unwords (map show outputs)

data Abs = Abs

instance Operation Abs LaxTensorProperties where
    applyOperation Abs [t] = [t]
    applyOperation Abs _ = error "WrongNumberInputsError"
    symbol Abs _ = "abs"

data Add = Add

instance Operation Add LaxTensorProperties where
    applyOperation Add [x, _] = [x]
    applyOperation Add _ = error "WrongNumberInputsError"
    symbol Add _ = "add"

newtype Concatenate = Concatenate Int

instance Operation Concatenate LaxTensorProperties where
    applyOperation (Concatenate _) [] = error "WrongNumberInputsError"
    applyOperation (Concatenate d) (LaxTensorProperties s t : xs) = [LaxTensorProperties newshape t]
      where
        newshape = shapeConcat (map tensorType (LaxTensorProperties s t : xs)) d
        tensorType (LaxTensorProperties s' _) = s'
    symbol (Concatenate d) _ = "concatenate" ++ showAsParameters [("dimensions", d)]

showAsParameters :: (Show v, Show k) => [(k, v)] -> String
showAsParameters params = "[" ++ intercalate "," (map f params) ++ "]"
  where
    f (x, y) = show x ++ "=" ++ show y

type Axis = Int

type Shape = [Axis]

shapeSingleConcat :: Shape -> Shape -> Axis -> Shape
shapeSingleConcat (a : as) (b : bs) axis = f (a : as) (b : bs) axis 0
  where
    f :: Shape -> Shape -> Axis -> Axis -> Shape
    f (c : cs) (d : ds) target cur
        | target == cur =
            if cs == ds
                then c + d : cs
                else error "shape mismatch"
        | target > cur =
            if c == d
                then c : f cs ds target (cur + 1)
                else error "shape mismatch"
        | otherwise = error "Not enough axes"
    f _ _ _ _ = error "Should not happen"
shapeSingleConcat _ _ _ = error "Shape should be non empty list"

shapeConcat :: [Shape] -> Axis -> Shape
shapeConcat (shap : shapes) axis = foldl (\x y -> shapeSingleConcat x y axis) shap shapes
shapeConcat [] _ = error "no shapes to concat"
