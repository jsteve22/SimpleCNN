#!/usr/bin/env python
# Date: 7/18/2022
# Create a function to perform NTT transform 
# and convolution multiplication

from custom_bfv.poly import Poly
import pickle as pkl
from random import randint
import random

class NTT(object):

	def __init__(self,n,M=2**5,N=2**5):
		# define the size of the length of vector
		self.n = n
		self.M = M
		self.N = N
		# self.find_working_mod(M)
		self.invn = self.extended_euclidean( self.n, self.N )
		# self.generate_params()

	def generate_params(self):
		# this function will generate working
		# parameters for NTT
		self.find_generator()
		self.calc_prim_root()

		self.invpsi = self.extended_euclidean( self.psi, self.N )
		self.invw = self.extended_euclidean( self.w, self.N )

	def find_working_mod(self, M):
		# find the working modulus for the
		# class. This will be a prime number
		# greater or equal to M

		k = (self.M // self.n) + 1
		N = (self.n * k) + 1
		while self.is_prime(N) == False:
			k += 1
			N = (self.n * k) + 1
		self.N = N
		self.k = k
		return
	
	'''
	def is_prime(self, n):
		if n == 2 or n == 3: return True
		if n < 2 or n%2 == 0: return False
		if n < 9: return True
		if n%3 == 0: return False
		r = int(n**0.5)
		# since all primes > 3 are of the form 6n ± 1
		# start with f=5 (which is prime)
		# and test f, f+2 for being prime
		# then loop by 6. 
		f = 5
		while f <= r:
			if n % f == 0: return False
			if n % (f+2) == 0: return False
			f += 6
		return True
	'''

	def is_prime(self, n, k=10):
		if n == 2 or n == 3: return True
		if n < 2 or n%2 == 0: return False
		if n < 9: return True
		if n%3 == 0: return False

		s, d = 0, n-1
		while d % 2 == 0:
			s += 1
			d //= 2
		
		for _ in range(k):
			a = random.randrange(2, n-1)
			if self.check_composite(a, d, s, n):
				return False
		return True

	def check_composite(self, a, d, s, n):
		x = pow(a, d, n)
		if x == 1 or x == n - 1:
			return False
		for _ in range(s-1):
			x = pow(x, 2, n)
			if x == n - 1:
				return False
		return True

	def find_generator(self):
		# this will find a generator which
		# would help calculate a primitive 
		# root of unity
		
		# find factors of N-1
		return

	def calc_prim_root(self):
		# this will calculate the primitive
		# root of unity needed to convert
		# into NTT
		# self.psi = (self.g ** self.k) % self.N
		# self.w = (self.g ** (self.k*2)) % self.N
		for w in range(2, self.N):
			if pow(w, 2*self.n, self.N) == 1 and pow(w, self.n, self.N) != 1:
				self.psi = w
				self.w = (w*w) % self.N
				break

	def extended_euclidean(self, a, n):
		# this function will calculate the inverse of a mod n
		# such that t*a === 1 mod n
		# 
		# this will be adapted from pseudo-code on wikipedia

		t = 0
		newt = 1
		r = n
		newr = a

		while newr != 0:
			quo = r // newr
			t, newt = newt, (t - quo*newt)
			r, newr = newr, (r - quo*newr)

		if t < 0:
			t += n

		return t

	def inverse_mod(self, a, n):
		# this function will find the inverse of a such
		# that t*a === 1 mod n
		# 
		# n must be a prime number
		
		p = n

		t = (a**(p-2))%p
		return t

	def NTT(self, p: Poly):
		# this will convert p into NTT

		if len(p) > self.n:
			raise ValueError("Input too large for NTT")

		psi = [0]*self.n

		for i in range(self.n):
			psi[i] = (self.psi ** self.bitrev(i)) % self.N

		a = p.copy()
		# ret = Poly( ret.poly + ([0]*(self.n-len(p))) )

		m = 1
		k = self.n // 2
		while m < self.n:
			for i in range(m):
				jFirst = 2 * i * k
				jLast = jFirst + k
				# wi = psi[ self.bitrev(m+i) ]
				wi = psi[ m+i ]
				for j in range(jFirst,jLast):
					# wrev = ( (self.w ** self.bitrev(m+i)) % self.N )
					l = j + k
					t = a[j]
					u = a[l] * wi
					a[j] = (t + u) % self.N
					a[l] = (t - u) % self.N

			m = m * 2
			k = k//2

		return a 

	def iNTT(self, p: Poly):
		# convert inverse NTT

		if len(p) > self.n:
			raise ValueError("Input too large for iNTT")

		invpsi = [0]*self.n

		for i in range(self.n):
			invpsi[i] = (self.invpsi ** self.bitrev(i)) % self.N

		# ret = Poly([0]*self.n)
		a = p.copy()
		# a = Poly( a.poly + ([0]*(self.n-len(p))) )

		m = self.n // 2
		k = 1
		while m >= 1:
			for i in range(m):
				jFirst = 2 * i * k
				jLast = jFirst + k
				# wi = psi[ self.bitrev(m+i) ]
				wi = invpsi[ m+i ]
				for j in range(jFirst,jLast):
					l = j + k
					t = a[j]
					u = a[l]
					a[j] = ( t + u ) % self.N
					a[l] = (( t - u ) * wi) % self.N
			
			m = m // 2
			k = k * 2

		for i in range(self.n):
			a[i] = a[i] * self.invn
			a[i] = a[i] % self.N

		return a

	def conv_mult(self, x: Poly, y: Poly):
		# compute the circular convolutional
		# multiplication of x and y

		z = Poly([0]*self.n)

		for i in range(self.n):
			z[i] = ( x[i] * y[i] ) % self.N

		return z 

	def bitrev(self, k: int):
		# return bit-reversed order of k
		
		revk = 0
		bit_len = (self.n-1).bit_length()

		for i in range( bit_len ):
			revk += (2 ** (bit_len-1-i) ) * ( ((2**i) & k ) >> i )

		return revk

	def print_root_unity(self, i=None):
		# print the powers of psi mod N

		if (i == None):
			i = 2*self.n

		for j in range(1,i+1):
			x = self.psi ** j
			if (x%self.N) == 1:
				print(f'{i}: {x%self.N}')
		

def main():
	sz = 2**10
	ntt = NTT(sz,2**20)

	# a = Poly([6,0,10,7])

	arr = []
	brr = []
	for i in range(sz):
		arr.append( randint(0,ntt.N) )
		brr.append( randint(0,ntt.N) )

	a = Poly([1,2,3,4,5,6,7,8]+([0]*2040))
	b = Poly([1,1,1,1,1,1,1,1]+([0]*2039)+[4])
	a = Poly( arr )
	b = Poly( brr )
	# a = Poly([1,2,3,4,5,6,7,8])


	# b = Poly([2,1,3,4,1,3,5,2])
	print(f'a: {a}')
	print(' ')

	pre_a = a.copy()

	pre_a = pre_a % ntt.N

	# n_a = ntt.NTT( pre_a )
	merge_a = ntt.merge_NTT( a )
	merge_b = ntt.merge_NTT( b )
	merge_c = ntt.conv_mult( merge_a, merge_b )
	# nb = ntt.NTT( b )

	# print(f'n_a:     {n_a}')
	print(f'merge_a: {merge_a}')
	print(' ')

	# ra = ntt.iNTT( n_a )
	mra = ntt.merge_iNTT( merge_a )
	mrc = ntt.merge_iNTT( merge_c )

	# print(f'ra: {ra}')
	print(f'mra: {mra}')
	# print(f'mrc: {mrc}')

	print(f'mra==a: {mra==a}')

	# polynomial multiplication
	c = a * b
	fn = Poly( [1] + ([0]*(sz-1)) + [1] )
	quo,c = c // fn
	c = c % ntt.N
	# print(c)

	print(f'mrc==c: {mrc==c}')
	

def test_addition():
	sz = 2**5
	ntt = NTT(sz,2**15)

	# a = Poly([6,0,10,7])

	arr = []
	brr = []
	for i in range(sz):
		arr.append( randint(0,ntt.N) )
		brr.append( randint(0,ntt.N) )

	a = Poly( arr )
	b = Poly( brr )

	merge_a = ntt.merge_NTT( a )
	merge_b = ntt.merge_NTT( b )

	merge_c = merge_a + merge_b

	mc = ntt.merge_iNTT( merge_c )

	c = a + b
	c = c % ntt.N

	print(f'mc: {mc}')
	print(f'c:  {c}')
	print(f'c==mc: {c==mc}')

def mult():
	sz = 2**10
	ntt = NTT(sz,2**38)

	arr = []
	for i in range(sz):
		arr.append( i )

	a = Poly( arr )

	ntt_a = ntt.NTT( a )

	res = ntt.conv_mult(ntt_a, ntt_a)

	res = ntt.iNTT( res )
	res = res % ntt.N

	b = a * a
	print(f'ntt.N = {ntt.N}')
	print(f'ntt.psi = {ntt.psi}')
	print(f'ntt.M = {ntt.M}')
	print(f'ntt.n = {ntt.n}')
	print(f'ntt.k = {ntt.k}')
	b = b % ntt.N
	fn = Poly([1] + ([0]*(sz-1)) + [1])

	q, rem = b // fn
	rem = rem % ntt.N

	# print(f'ntt version: {res}')
	# print(f'normal:      {rem}')
	print(res == rem)


if __name__ == '__main__':
	# main()
	# test_addition()
	mult()

	pass
