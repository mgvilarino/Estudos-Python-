#!/bin/bash

# Script para calcular el interés simple

echo "Ingrese el capital (P):"
read principal

echo "Ingrese la tasa de interés anual (R) en porcentaje:"
read rate

echo "Ingrese el tiempo (T) en años:"
read time

# Cálculo del interés simple: SI = (P × R × T) / 100
interest=$(echo "$principal * $rate * $time / 100" | bc)

echo "----------------------------------"
echo "El interés simple es: $interest"
echo "----------------------------------"
