{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "\n",
    "u = np.array([3,4])\n",
    "v = np.array([30,40])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5.0\n",
      "50.0\n"
     ]
    }
   ],
   "source": [
    "# length / magnitude of a vector\n",
    "\n",
    "print(np.linalg.norm(u))\n",
    "print(np.linalg.norm(v))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "def direction(x):\n",
    "    return x/np.linalg.norm(x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.6 0.8]\n",
      "[0.6 0.8]\n"
     ]
    }
   ],
   "source": [
    "w = direction(u)\n",
    "z = direction(v)\n",
    "print(w)\n",
    "print(z)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1.0\n",
      "1.0\n"
     ]
    }
   ],
   "source": [
    "# Direction vector always has a unit length i.e. norm / magnitude of the direction vector is always 1 \n",
    "\n",
    "print(np.linalg.norm(w))\n",
    "print(np.linalg.norm(z))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Dot product / projection of one vector on other, influence of one over the other\n",
    "import math\n",
    "\n",
    "def geometric_dot_product(x,y, theta):\n",
    "    x_norm = np.linalg.norm(x)\n",
    "    y_norm = np.linalg.norm(y)\n",
    "    return x_norm * y_norm * math.cos(math.radians(theta))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "34.00000000000001\n"
     ]
    }
   ],
   "source": [
    "\n",
    "\n",
    "theta = 45   # reduce theta, dot product will increase\n",
    "X = [3,5]\n",
    "y = [8,2]\n",
    "\n",
    "print(geometric_dot_product(X, y, theta))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "def dot_product(X,y):\n",
    "    result = 0\n",
    "    for i in range (len(X)):\n",
    "        result = result + X[i]*y[i]\n",
    "    return(result)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "34\n"
     ]
    }
   ],
   "source": [
    "X = [3,5]\n",
    "y = [8,2]\n",
    "\n",
    "print(dot_product(X,y))  # should get the same result as geometric dot produt. \n",
    "                         # instead of finding the theta, simply multiply the magnitudes of the respective dimensions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.9999999999999997\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Content_ACER\\Anaconda3\\lib\\site-packages\\ipykernel_launcher.py:4: FutureWarning: `rcond` parameter will change to the default of machine precision times ``max(M, N)`` where M and N are the input matrix dimensions.\n",
      "To use the future default and silence this warning we advise to pass `rcond=None`, to keep using the old, explicitly pass `rcond=-1`.\n",
      "  after removing the cwd from sys.path.\n"
     ]
    }
   ],
   "source": [
    "X = np.array([0, 1, 2, 3])\n",
    "y = np.array([-1, 0.2, 0.9, 2.1])\n",
    "A = np.vstack([X, np.ones(len(X))]).T\n",
    "m, c = np.linalg.lstsq(A, y)[0]\n",
    "print(m)  # recall m is tan(@) and very close to 1 indicating the theta to be 45 degrees"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAX8AAAD8CAYAAACfF6SlAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvNQv5yAAAIABJREFUeJzt3XmYVNW19/HvkoCoEFQgDoiCCVEGAaVVCE5XxKCJ4hs1wdyIQxQ1YMLV6CV6HWKMAwZEEcEWCBAURQZBZBAEQVGRVkAmxQ7B0IFoi4oSZGh6vX/s0nSabrqhquvUqfp9nqceatj0WSdlFqv32Wdtc3dERCS37Bd1ACIikn5K/iIiOUjJX0QkByn5i4jkICV/EZEcpOQvIpKDlPxFRHKQkr+ISA5S8hcRyUHfijqAyjRq1MibNWsWdRgiIrHy9ttvf+Lujasal7HJv1mzZhQUFEQdhohIrJjZh9UZp2kfEZEcpOQvIpKDlPxFRHJQxs75V2Tnzp0UFRWxbdu2qENJqbp163LUUUdRu3btqEMRkRwRq+RfVFRE/fr1adasGWYWdTgp4e5s2rSJoqIimjdvHnU4IpIjYjXts23bNho2bJg1iR/AzGjYsGHW/TYjIpktVskfyKrE/7VsPCcRyWyxS/4iItls6lQYMaLmj5N08jezpmY2z8xWm9lKM/tNBWPMzB41s0Ize9fMTkr2uFGpV68eABs2bOCSSy6JOBoRyRYffww9ekD37iH5l5bW7PFSUfmXADe7e0ugI9DbzFqVG3Me0CLx6AUMTcFxI3XkkUcyYcKEqMMQkZhzh7FjoWVLmDwZ7r0X5s+H/Wp4XibpH+/uG939ncTzL4HVQJNyw7oDYzx4EzjYzI5I9thRWrduHW3atAFg1KhR/OQnP6Fbt260aNGCW2+99ZtxL730Ep06deKkk07i0ksvZcuWLVGFLCIZZv16+PGP4fLL4bjjYOlSuP12SMeq75Qu9TSzZsCJwKJyHzUB1pd5XZR4b+O+Hqtv3/A/VCq1bw+DBu3b3126dClLlixh//3357jjjuPGG2/kgAMO4N5772XOnDkcdNBBPPjggwwcOJA777wztYGLSKyUlsITT8Ctt4bnjzwCvXtDrVrpiyFlyd/M6gETgb7u/kX5jyv4K17Bz+hFmBbi6KOPTlVoadGlSxcaNGgAQKtWrfjwww/5/PPPWbVqFZ07dwZgx44ddOrUKcowRSRia9bANdfAq6/COedAfj5EcYtPSpK/mdUmJP6n3H1SBUOKgKZlXh8FbCg/yN3zgXyAvLy83f5xKGtfK/Sasv/++3/zvFatWpSUlODudO3alXHjxkUYmYhkgpISGDgQ7roL6taFkSPhyishqpXeqVjtY8AIYLW7D6xk2FSgZ2LVT0dgs7vv85RPXHTs2JGFCxdSWFgIwNatW1mzZk3EUYlIui1bBqeeCv/7v3DeebBqFVx1VXSJH1JT+XcGLgeWm9nXs/C3AUcDuPswYDpwPlAIbAWuSsFxM17jxo0ZNWoUl112Gdu3bwfg3nvv5fvf/37EkYlIOmzfHlbvPPAANGwIEybAxRdHHVVg7nucXYlMXl6el9/MZfXq1bRs2TKiiGpWNp+bSC56/XX45S/hvffgiivClM+hh9b8cc3sbXfPq2qc7vAVEUmhLVvgN7+B006DrVth5kwYNSo9iX9vKPmLiKTI7Nlwwgnw6KNh6eaKFfDDH0YdVcVil/wzdZoqGdl4TiK55LPP4Oqr4dxzYf/9wzLOwYOhfv2oI6tcrJJ/3bp12bRpU1Yly6/7+detWzfqUERkH0yaBK1awZgx8LvfhZtPTzst6qiqFqvNXI466iiKioooLi6OOpSU+nonLxGJj3/+E/r0gYkTQ3eA6dPhxBOjjqr6YpX8a9eurd2uRCRS7qHK/5//CRd077sPfvvb9PTjSaVYJX8RkSh9+CFcdx3MmgWdO8Pw4XD88VFHtW9iNecvIhKF0lJ47DFo3RoWLgzPFyyIb+IHVf4iInv03nuhEdvChWHZ5hNPwDHHRB1V8lT5i4hUYOfOMJ/frl3oxTN6NMyYkR2JH1T5i4jsZsmSsG5/6VK45JIwzXPYYVFHlVqq/EVEErZtC2v1Tz45LOWcOBGeey77Ej+o8hcRAeC110IjtjVrQtX/pz/BIYdEHVXNUeUvIjntyy/DzVqnnw47doT+PCNGZHfiByV/EclhM2dCmzbw+OOhE+fy5WFrxVyg5C8iOWfTptBj/7zz4KCDwjLOQYOgXr2oI0sfJX8RyRnuYTetVq3g6afh//4vrOzp1CnqyNJPF3xFJCds3Bh67E+eDB06wEsvhTX8uUqVv4hkNXcYOTJU+zNmQP/+8OabuZ34QZW/iGSxv/0NevWCOXPgjDPgySfh+9+POqrMoMpfRLLOrl3wyCNhJc+iRTB0KMybp8Rflip/Eckqq1aFRmxvvBFW8zzxBDRtGnVUmUeVv4hkhZ074d57w25aa9bA2LHw4otK/JVR5S8isVdQEFozvPsu9OgRpny+852oo8psKan8zWykmX1sZisq+fwsM9tsZksTjztTcVwRyW1ffQW33gqnngqffAJTpsC4cUr81ZGqyn8U8BgwZg9jXnX3H6foeCKS4+bPD3P7hYVw7bVhCefBB0cdVXykpPJ39wXAp6n4WSIie/LFF3DDDXDWWWF7xZdfhvx8Jf69lc4Lvp3MbJmZzTCz1hUNMLNeZlZgZgXFxcVpDE1E4mD69LCPbn4+3HRTmOM/++yoo4qndCX/d4Bj3L0dMBh4vqJB7p7v7nnunte4ceM0hSYime6TT+AXv4Af/QgaNIDXX4cBA0JTNtk3aUn+7v6Fu29JPJ8O1DazRuk4tojElzs88wy0bAnjx8Ndd8E774QLvJKctCR/MzvczCzx/JTEcTel49giEk//+AdcdBFcdhk0bw5vvw133w116kQdWXZIyWofMxsHnAU0MrMi4C6gNoC7DwMuAW4wsxLgK6CHu3sqji0i2cUdhg+H3/423Lj1pz9B375Qq1bUkWWXlCR/d7+sis8fIywFFRGp1F//GpZtzpsXVvM8+SR873tRR5Wd1N5BRCK3axcMHAgnnBCmd/LzYe5cJf6apPYOIhKpFStCa4a33oILLggdOJs0iTqq7KfKX0QisWMH/P73cNJJsHZtaMswZYoSf7qo8heRtHvrrVDtr1gBP/95aMTWSIu/00qVv4ikzdatYRVPp07w2Wfwwgvw1FNK/FFQ5S8iaTFvXmjEtnYtXH89PPBAuFtXoqHKX0Rq1ObNYR/ds8+G/faDV14JF3WV+KOl5C8iNeaFF6BVKxgxAm65BZYtgzPPjDoqASV/EakBxcWhLcOFF0LDhmET9f794cADo45MvqbkLyIp4w5PPx0asU2cCPfcE7ZYzMuLOjIpTxd8RSQl1q8Pm6y8+CJ07Bj687SucOcOyQSq/EUkKaWlMGxYSPTz5sGgQfDaa0r8mU6Vv4jssw8+CI3Y5s+HLl1CT55jj406KqkOVf4istdKSuChh6BtW1i6NKzmmT1biT9OVPmLyF55993QmqGgALp3h8cfhyOPjDoq2Vuq/EWkWrZvhzvvhA4d4O9/D9sqTp6sxB9XqvxFpEpvvBGq/dWroWfP0Hu/YcOoo5JkqPIXkUr9619hC8XOnWHLFpg+HUaPVuLPBqr8RaRCc+aElTzr1kHv3nD//VC/ftRRSaqo8heR//D552GKp2tXqF0bFiyAxx5T4s82Sv4i8o3nnw+N2EaPhn79QiO200+POiqpCZr2ERE++ghuvBGeew7at4dp08L2ipK9VPmL5DB3GDMmNGKbMgX++MewxaISf/ZT5S+So/7+d7juOpg5E37wg3CX7vHHRx2VpEtKKn8zG2lmH5vZiko+NzN71MwKzexdM1NdIRKR0lIYMiQ0Xnv1VXj00XBRV4k/t6Rq2mcU0G0Pn58HtEg8egFDU3RcEdkL778fdtLq0ydU+ytWhLn+WrWijkzSLSXJ390XAJ/uYUh3YIwHbwIHm9kRqTi2iFRt586wYXq7drByJYwaFaZ7mjWLOjKJSrrm/JsA68u8Lkq8t7HsIDPrRfjNgKOPPjpNoYlktyVLwrr9JUvg4ovDmv3DD486Kolaulb7WAXv+W5vuOe7e5675zVu3DgNYYlkr23b4Pbb4eSTYcMGmDAhPJT4BdJX+RcBTcu8PgrYkKZji+SchQtDtf/++3DllTBgABx6aNRRSSZJV+U/FeiZWPXTEdjs7hur+ksisne2bIFf/zrclbttG8yaBX/+sxK/7C4llb+ZjQPOAhqZWRFwF1AbwN2HAdOB84FCYCtwVSqOKyL/9tJL0KtXWL/fpw/cdx/Uqxd1VJKpUpL83f2yKj53oHcqjiUi/+nTT+Hmm8MKnuOPD2v3O3eOOirJdGrvIBJjEyeGRmx/+Uu4uLtkiRK/VI/aO4jE0MaNYWpn0iQ48cSwZr99+6ijkjhR5S8SI+5heqdVK3jxxXDj1ltvKfHL3lPlLxIT69aFC7qzZ8Npp8Hw4XDccVFHJXGlyl8kw5WWwuDB0KZN2Eh9yBCYP1+JX5Kjyl8kg61eDddcA6+/Dt26wbBhcMwxUUcl2UCVv0gG2rkzbKzSvj28917YcGX6dCV+SR1V/iIZ5p134Oqrw/65P/1p6Ld/2GFRRyXZRpW/SIb46quwafopp4Q9dSdPhmefVeKXmqHKXyQDvPpqmNtfsyY0ZHvoITjkkKijkmymyl8kQl9+Cb17wxlnwI4dYRnn8OFK/FLzlPxFIjJjRthHd+hQ6Ns3bKl4zjlRRyW5QslfJM02bYKePeH880PXzYUL4eGH4aCDoo5McomSv0iauMP48dCyJYwbB3fcERqxdeoUdWSSi3TBVyQNNmwIc/vPPw8dOsCcOdC2bdRRSS5T5S9Sg9xhxIjQiG3mTOjfH958U4lfoqfKX6SGrF0bGrG9/HJYzTN8OLRoEXVUIoEqf5EU27ULBg2CE04I7ZaHDoV585T4JbOo8hdJoZUrw01aixaF1TzDhkHTplFHJbI7Vf4iKbBjB/zhD2FXrcJCGDsWpk1T4pfMpcpfJEmLF4dqf/ly6NEDHnkEvvOdqKMS2TNV/iL7aOtWuPVW6Ngx3Lg1ZUpYv6/EL3Ggyl9kH8yfHxqxFRbCtdeGRmwNGkQdlUj1qfIX2QubN8P118NZZ4XtFV9+GfLzlfglflKS/M2sm5m9b2aFZtavgs+vNLNiM1uaeFyTiuOKpNOLL4ZGbE8+CTfdFOb4zz476qhE9k3S0z5mVgsYAnQFioDFZjbV3VeVG/qsu/dJ9ngi6VZcHLpuPv10SP4TJ8Kpp0YdlUhyUlH5nwIUuvtad98BPAN0T8HPFYmUOzzzTGjN8NxzcNddYYtFJX7JBqlI/k2A9WVeFyXeK+9iM3vXzCaYWYWrn82sl5kVmFlBcXFxCkIT2Tf/+Ad07w6XXQbNm8Pbb8Pdd0OdOlFHJpIaqUj+VsF7Xu71C0Azd28LzAFGV/SD3D3f3fPcPa9x48YpCE1k77iHOf1WrULnzQED4I03QqsGkWySiuRfBJSt5I8CNpQd4O6b3H174uWTQIcUHFckpQoLoUuX0IytQ4dwQfemm6BWragjE0m9VCT/xUALM2tuZnWAHsDUsgPM7IgyLy8EVqfguCIpsWtXqPDbtg3TO/n5YQnnd78bdWQiNSfp1T7uXmJmfYBZQC1gpLuvNLN7gAJ3nwr82swuBEqAT4Erkz2uSCqsWAFXXx1aNFxwQejA2aSiK1YiWcbcy0/PZ4a8vDwvKCiIOgzJUjt2wH33hUeDBjB4MPzsZ2AVXcESiREze9vd86oap/YOknPeeitU+ytXwn//d+i936hR1FGJpJfaO0jO2LoVbr45bJi+eXNouTx2rBK/5CZV/pIT5s4NDdjWrg29eR58EL797aijEomOKn/Jap9/HpJ+ly6w337wyivhoq4Sv+Q6JX/JWlOnhl48I0fCLbfAsmVw5plRRyWSGZT8Jet8/HHYUat7d2jYMOyn278/HHhg1JGJZA4lf8ka7vDUU6E1w6RJcM89UFAAeVUuehPJPbrgK1lh/Xq44YbQc79jRxg+PEz5iEjFVPlLrJWWwrBhIdHPmxfW7L/2mhK/SFVU+UtsffBB2Ed3wYKwmic/H449NuqoROJBlb/ETklJuIDbtm1YwTNiBMyercQvsjdU+UusLFsGv/xl6L7ZvTs8/jgceWTUUYnEjyp/iYXt2+GOO8LKnfXrYfx4mDxZiV9kX6nyl4z3xhuh2l+9Gi6/HB5+OKzfF5F9p8pfMtaWLdC3L3TuHJ5Pnw5jxijxi6SCKn/JSLNnh+0U162D3r3h/vuhfv2ooxLJHqr8JaN89lmY4jn3XKhdOyzjfOwxJX6RVFPyl4wxeXJozTB6NPTrF1b2nH561FGJZCdN+0jkPvoIbrwRnnsO2rULm6x06BB1VCLZTZW/RMY9XMBt2RKmTIE//jFspK7EL1LzVPlLJD78MOyoNXNm2FZxxIjwj4CIpIcqf0mr0lIYMgTatIFXX4VHHw1/KvGLpJcqf0mb998Pjdheew26dg2N2Jo1izoqkdykyl9q3M6d8MAD4WLuihXw5z/DrFlK/CJRSknyN7NuZva+mRWaWb8KPt/fzJ5NfL7IzJql4riS+ZYsgVNPhd/9Dn70o9Ci4corwSzqyERyW9LJ38xqAUOA84BWwGVm1qrcsF8Cn7n794CHgQeTPa5ktm3b4Pbb4eSTYcMGmDABJk6Eww+POjIRgdRU/qcAhe6+1t13AM8A3cuN6Q6MTjyfAHQxU+2XrRYuhPbt4b774Be/gFWr4OKLo45KRMpKRfJvAqwv87oo8V6FY9y9BNgM7Naey8x6mVmBmRUUFxenIDRJpy+/DDdrnX56qPxnzoRRo+DQQ6OOTETKS0Xyr6iC930Yg7vnu3ueu+c1btw4BaFJusyaFZZvDhkCffqEC7s//GHUUYlIZVKR/IuApmVeHwVsqGyMmX0LaAB8moJjS8Q+/TRcwO3WDQ444N9r9+vVizoyEdmTVCT/xUALM2tuZnWAHsDUcmOmAlcknl8CzHX33Sp/iZeJE0MjtrFj4bbbYOnS0HtfRDJf0jd5uXuJmfUBZgG1gJHuvtLM7gEK3H0qMAL4i5kVEir+HskeV6KzcWOY2pk0CU48Mcztt28fdVQisjdScoevu08Hppd7784yz7cBl6biWBId93AB96ab4KuvwgYrN98c+u6LSLyovYNUy7p1YWet2bPhtNNg+HA47riooxKRfaX2DrJHu3aFC7ht2oSN1IcMgfnzlfhF4k6Vv1Rq9erQiO3118NqnmHD4Jhjoo5KRFJBlb/sZufOsLFK+/bw3nthw5Xp05X4RbKJKn/5D2+/HTZQX7YMLr0UBg+Gww6LOioRSTVV/gKE1Tv9+oUOnB99FJZxjh+vxC+SrVT5CwsWhLn9Dz4IVf9DD8Ehh0QdlYjUJFX+OeyLL6B3bzjzzDDPP3t2WMKpxC+S/ZT8c9SMGWH55tCh0LdvaMR2zjlRRyUi6aLkn2M2bYKePeH880PztYUL4eGH4aCDoo5MRNJJyT9HuIcLuC1bwrhxcMcdYYvFTp2ijkxEoqALvjlgwwb41a9gyhTo0AHmzIG2baOOSkSipMo/i7nDiBGh7fKsWdC/P7z5phK/iKjyz1pr18K118LcuXDGGWEVT4sWUUclIplClX+W2bULBg2CE06AxYvDap5585T4ReQ/qfLPIitXhpu0Fi0Kq3mGDYOmTav+eyKSe1T5Z4EdO+Cee8KuWoWF8NRTMG2aEr+IVE6Vf8wtXhyq/eXLoUeP0Hu/ceOooxKRTKfKP6a2boVbboGOHcONW1OmhPX7SvwiUh2q/GPolVfCSp7CwvDnQw9BgwZRRyUicaLKP0Y2b4brr4f/+i8oLYWXX4b8fCV+Edl7Sv4xMW0atG4NTz4JN90U5vjPPjvqqEQkrpT8M1xxMfz853DBBXDwwWE/3QED4MADo45MROJMyT9DuYcLuK1awYQJcPfd8M47YactEZFkJZX8zexQM5ttZh8k/qxwGxAz22VmSxOPqckcMxcUFcGFF4aK/9hjQ9K/6y6oUyfqyEQkWyRb+fcDXnb3FsDLidcV+crd2yceFyZ5zKxVWhou4LZuHS7mDhgQpnnatIk6MhHJNskm/+7A6MTz0cBFSf68nFVYCF26wHXXhbbLy5eHC7u1akUdmYhko2ST/2HuvhEg8ed3KhlX18wKzOxNM6v0Hwgz65UYV1BcXJxkaPFQUhIq/LZtw/ROfn6o+r/73agjE5FsVuVNXmY2Bzi8go9u34vjHO3uG8zsWGCumS1397+WH+Tu+UA+QF5enu/Fz4+l5ctDa4bFi8NqnqFDoUmTqKMSkVxQZfJ390q39Tazj8zsCHffaGZHAB9X8jM2JP5ca2avACcCuyX/XLF9O9x3X3gccgg88wz89KdgFnVkIpIrkp32mQpckXh+BTCl/AAzO8TM9k88bwR0BlYledzYWrQozOnfcw/87GewalX4U4lfRNIp2eT/ANDVzD4AuiZeY2Z5ZjY8MaYlUGBmy4B5wAPunnPJ/1//ChdwO3UKbRqmTYOxY6FRo6gjE5FclFRjN3ffBHSp4P0C4JrE89eBE5I5TtzNnRsasK1dG3rzPPggfPvbUUclIrlMd/jWoM8/D0m/SxfYb7/QjXPoUCV+EYmekn8NmTIltGYYORJuvRXefRfOPDPqqEREAiX/FPv447Cj1kUXhfn8RYvCNM8BB0QdmYjIvyn5p4h7uIDbsiVMngx/+AMUFEBeXtSRiYjsTjt5pcD69eFC7vTpYVvFESPClI+ISKZS5Z+E0tJwAbd163Axd9AgeO01JX4RyXyq/PfRmjVhJc+CBXDOOaEnT/PmUUclIlI9qvz3UkkJ9O8P7drBsmVhiuell5T4RSReVPnvhWXL4OqrQ/fNiy6CIUPgyCOjjkpEZO+p8q+G7dvhjjvCyp2iIhg/HiZNUuIXkfhS5V+FN94IbZdXr4aePWHgQGjYMOqoRESSo8q/Elu2QN++0LlzaMo2YwaMHq3ELyLZQZV/BWbPhl69YN066N0b7r8f6tePOioRkdRR5V/GZ5+FC7rnngt16oRlnI89psQvItlHyT9h8uRwc9aYMdCvX1jZc/rpUUclIlIzcn7a55//hBtvhAkToH17ePFFOOmkqKMSEalZOVv5u4cqv1UreOGFsJ/uW28p8YtIbsjJyv/DD+G662DWLPjBD8JduscfH3VUIiLpk1OVf2lpuIDbunVowDZ4MLz6qhK/iOSenKn8338/3Ky1cGFYzfPEE9CsWdRRiYhEI+sr/507wzr9du1g1SoYNQpmzlTiF5HcltWV/5IlodpfsgQuvjhM+Rx+eNRRiYhELysr/23b4Lbb4OSTYcOGsIxzwgQlfhGRr2Vd5f+3v8F554U5/quuggED4JBDoo5KRCSzJFX5m9mlZrbSzErNrNKtys2sm5m9b2aFZtYvmWNWpUkT+N73wjLOkSOV+EVEKpJs5b8C+AnwRGUDzKwWMAToChQBi81sqruvSvLYFapTB6ZNq4mfLCKSPZJK/u6+GsDM9jTsFKDQ3dcmxj4DdAdqJPmLiEjV0nHBtwmwvszrosR7uzGzXmZWYGYFxcXFaQhNRCQ3VVn5m9kcoKJ1Mre7+5RqHKOiXwu8ooHung/kA+Tl5VU4RkREkldl8nf3c5I8RhHQtMzro4ANSf5MERFJQjqmfRYDLcysuZnVAXoAU9NwXBERqUSySz3/n5kVAZ2AF81sVuL9I81sOoC7lwB9gFnAamC8u69MLmwREUlGsqt9JgOTK3h/A3B+mdfTgenJHEtERFInK9s7iIjInpl7Zi6qMbNi4MMkfkQj4JMUhROlbDkP0Llkqmw5l2w5D0juXI5x98ZVDcrY5J8sMytw90pbTsRFtpwH6FwyVbacS7acB6TnXDTtIyKSg5T8RURyUDYn//yoA0iRbDkP0Llkqmw5l2w5D0jDuWTtnL+IiFQumyt/ERGpRKyTf1WbxJjZ/mb2bOLzRWbWLP1RVk81zuVKMys2s6WJxzVRxFkVMxtpZh+b2YpKPjczezRxnu+a2UnpjrG6qnEuZ5nZ5jLfyZ3pjrE6zKypmc0zs9WJzZd+U8GYWHwv1TyXuHwvdc3sLTNbljiX31cwpuZymLvH8gHUAv4KHAvUAZYBrcqN+RUwLPG8B/Bs1HEncS5XAo9FHWs1zuUM4CRgRSWfnw/MIHR77QgsijrmJM7lLGBa1HFW4zyOAE5KPK8PrKngv69YfC/VPJe4fC8G1Es8rw0sAjqWG1NjOSzOlf83m8S4+w7g601iyuoOjE48nwB0sSp2nolIdc4lFtx9AfDpHoZ0B8Z48CZwsJkdkZ7o9k41ziUW3H2ju7+TeP4locdW+T01YvG9VPNcYiHxv/WWxMvaiUf5i7A1lsPinPyrs0nMN2M8NJjbDDRMS3R7p7ob3lyc+JV8gpk1reDzOKj25j4x0Snxa/sMM2sddTBVSUwbnEioMsuK3feyh3OBmHwvZlbLzJYCHwOz3b3S7yXVOSzOyb86m8RUeyOZiFUnzheAZu7eFpjDv6uBuInLd1Id7xBupW8HDAaejziePTKzesBEoK+7f1H+4wr+SsZ+L1WcS2y+F3ff5e7tCfucnGJmbcoNqbHvJc7JvzqbxHwzxsy+BTQgM3+Nr/Jc3H2Tu29PvHwS6JCm2FItazb3cfcvvv613UPn2tpm1ijisCpkZrUJyfIpd59UwZDYfC9VnUucvpevufvnwCtAt3If1VgOi3Pyr84mMVOBKxLPLwHmeuLKSYap8lzKzb9eSJjrjKOpQM/E6pKOwGZ33xh1UPvCzA7/ev7VzE4h/P9pU7RR7S4R4whgtbsPrGRYLL6X6pxLjL6XxmZ2cOL5AcA5wHvlhtVYDkuqn3+U3L3EzL7eJKYWMNLdV5rZPUCBu08l/EfyFzMrJPxr2SO6iCtXzXP5tZldCJQQzuXKyALeAzMbR1ht0cjCRj93ES5k4e7DCPs6nA8UAluBq6KJtGrVOJdLgBvMrAT4CuiRocVFZ+ByYHlifhkHMnpfAAAAWElEQVTgNuBoiN33Up1zicv3cgQw2sxqEf6BGu/u09KVw3SHr4hIDorztI+IiOwjJX8RkRyk5C8ikoOU/EVEcpCSv4hIDlLyFxHJQUr+IiI5SMlfRCQH/X/Q7ttBRkBnYgAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x282aeabccf8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "\n",
    "plt.plot(X, m*X + c, 'b', label='line')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Exercise 1 on equation of a line - given the data, find the slope and intercept and express the equation, draw the line"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2.0\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Mukesh\\Anaconda3\\lib\\site-packages\\ipykernel_launcher.py:6: FutureWarning: `rcond` parameter will change to the default of machine precision times ``max(M, N)`` where M and N are the input matrix dimensions.\n",
      "To use the future default and silence this warning we advise to pass `rcond=None`, to keep using the old, explicitly pass `rcond=-1`.\n",
      "  \n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "\n",
    "X = np.array([0, 1, 2, 3, 4])\n",
    "y = np.array([2,4,6,8,10])\n",
    "A = np.vstack([X, np.ones(len(X))]).T\n",
    "m, c = np.linalg.lstsq(A, y)[0]\n",
    "print(m)  # recall m is tan(@) and very close to 1 indicating the theta to be 45 degrees"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x282aeb25a20>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "\n",
    "plt.plot(X, m*X + c, 'b', label='line')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
