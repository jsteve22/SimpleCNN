#!/usr/bin/env python
# this program will be an attempt to create the bfv
# encrption scheme that is described in the BFV paper.

import numpy as np
from numpy.polynomial import polynomial as p
import random
import sys

import pdb
from poly import Poly

def main():
	# this main function is using the lpr() class
	# encryption scheme

	# create an encryption scheme instance
	bfv = BFV()

	# create plaintext you want to encrypt
	pt = 5

	# encrypt plaintext into ciphertext
	ct = bfv.encrypt(pt)

	# decrypt back into plaintext
	recovered_pt = bfv.decrypt(ct)

	# print results
	print(f'original pt: {pt}\trecovered pt: {recovered_pt}')
	print(f'{pt==recovered_pt}')
	print( bfv.opcount )

	return

class BFV():

	def __init__(self,q=2**15,t=2,n=2**4,std=2,fn=None,h=64,security=128,bitwidth=64,cache_size=1024,block_size=128,cache_hit_rate=0.9):
		"""
		this init method will initialize the important variables
		needed for this encryption scheme, including keys

		----- Variables -----
		q 	- Cipher Text Modulus
		t 	- Plain Text Modulus
		n 	- Degree of Ring Polynomial
		std - standard deviation for Gaussian distribution
		fn 	- Cyclotomic Polynomial from n
		
		----- Keys -----
		sk 	- secret key with hamming weight h=64
		pk 	- public key for encrypting messages
		rlk - relinearization key for reducing ciphertext size after multiplication 
		"""

		self.q = q
		self.t = t
		self.n = n
		self.std = std
		self.h = h
		self.sec = security
		# this will set the polynomial for the ring, if not declared then will be
		# the polynomial 1 + x^n
		self.fn = fn
		if (self.fn == None):
			self.fn = [1] + [0]*((n)-1) + [1]
		self.fn = Poly(self.fn)

		# create operation counters for different methods in BFV
		self.gen_counters()
		self.bitwidth = bitwidth

		# this will set the keys as none, but will then immediately generate
		# a public key, a private key, and a relinearization key
		self.sk = None
		self.pk = None
		self.rlk = None
		self.gen_keys()

		# save data for memory accessing
		self.cache_size = cache_size
		self.block_size = block_size
		self.cache_size_bytes = self.cache_size
		self.block_size_bytes = self.block_size
		self.cache_hit_rate = cache_hit_rate
		self.cache_miss_rate = 1 - cache_hit_rate

		self.cache_accesses = 0
		self.main_memory_accesses = 0


	def gen_keys(self):
		"""
		Generate keys, generate secret key first then public 
		and relinearization	keys afterwards
		"""
		self.gen_sk()
		self.gen_pk()
		
	def gen_sk(self):
		"""
		call the gen_binary_poly key to create a polynomial
		of only 0's and 1's for the secret key
		"""
		# self.sk = self.gen_binary_poly()
		# self.sk = self.gen_normal_poly()
		
		# set hamming weight of secret key
		sk = [1]*self.h
		sk = sk + [0]*(self.n-self.h)
		np.random.shuffle( sk )
		self.sk = Poly( sk )
	
		return

	def gen_pk(self):
		"""
		Generate public key from secret key and random polynomials.

		----- Random Polynomials -----
		a <- Uniform Distribution over q
		e <- Normal Distribution with (mean = 0, standard deviation = self.std)

		----- Calculation ----- 
		a = -(a*sk + e)

		----- Result -----
		pk = (b , a)

		"""
		if (self.sk == None):
			return
		# generate a uniformly distributed polynomial with coefs
		# from [0,q)
		a = self.gen_uniform_poly()

		# generate a normally distributed polynomial with integers
		# generated from a center of 0 and std of 2
		e = self.gen_normal_poly()

		# create a new polynomial _a which is -a
		_a = a * -1

		# then set e = -e
		e = e * -1

		# breakpoint()
		# create b from (-a * sk) - e
		b = self.polyadd( self.polymult(_a, self.sk), e)

		# set the public key to the tuple (b,a)
		# or (-[a*sk + e], a)
		self.pk = (b,a)
		return

	def gen_counters(self):
		"""
		this function will generate operation counters
		for encrypting, decrypting, key generation, 
		addition, multiplication, and relinearization
		"""

		# create operations counter object
		self.counters = {}
		self.counters['enc'] = self.default_counter()
		self.counters['dec'] = self.default_counter()
		self.counters['add'] = self.default_counter()

		self.counters['key'] = self.default_counter()
	
	def default_counter(self):
		counter = {}
		counter['add'] = 0
		counter['mul'] = 0
		counter['mod'] = 0
		return counter
	
	def encrypt(self,pt=0):
		"""
		encode plaintext integer into a plaintext polynomial
		and then into ciphertext polynomials

		----- Arguments -----
		pt	- Plain Text Integer

		----- Random Polynomials -----
		u 	<- Binary Distribution
		e1	<- Normal Distribution with (mean = 0, standard deviation = self.std)
		e2	<- Normal Distribution with (mean = 0, standard deviation = self.std)

		----- Calculation -----
		m = pt * (q / t)
		c0 = pk[0]*u + e1 + m
		c1 = pk[1]*u + e2
		ct = (c0, c1)

		----- Output -----
		ct	- Ciphertext of plain text
	
		"""

		counter = self.counters['enc']

		m = pt
		if ( type(pt) == int ):
			m = [pt]
			m = Poly(m)
		elif ( type(pt) == list ):
			m = Poly(m)

		m = m % self.q
		self.memory_access_polynomial(m)
		counter['mod'] += len(m)

		delta = self.q // self.t
		scaled_m = m.copy()
		scaled_m = (scaled_m * delta) 
		scaled_m = scaled_m  % self.q
		counter['mul'] += len(scaled_m)
		counter['mod'] += len(scaled_m)

		# create a new m, which is scaled my q//t % q
		# generated new error polynomials
		e1 = self.gen_normal_poly()
		e2 = self.gen_normal_poly()
		u = self.gen_binary_poly()

		# create c0 = pk[0]*u + e1 + scaled_m
		ct0 = self.polyadd( self.polyadd( self.polymult( self.pk[0], u), e1), scaled_m)
		self.memory_access_polynomial(u)
		self.memory_access_polynomial(e1)
		self.memory_access_polynomial(scaled_m)
		self.memory_access_polynomial(self.pk[0])
		counter['mul'] += len(self.pk[0])**2
		counter['add'] += len(self.pk[0])**2 + len(self.pk[0])*2
		counter['mod'] += len(self.pk[0])*3

		# create c1 = pk[1]*u + e2
		ct1 = self.polyadd( self.polymult( self.pk[1], u), e2)
		self.memory_access_polynomial(e2)
		self.memory_access_polynomial(u)
		self.memory_access_polynomial(self.pk[1])
		counter['mul'] += len(self.pk[1])**2
		counter['add'] += len(self.pk[1])**2 + len(self.pk[1])
		counter['mod'] += len(self.pk[1])*2

		return (ct0, ct1)

	def decrypt(self,ct):
		"""
		decrypt the cipher text to get the plaintext equivalent

		----- Arguments -----
		ct	- Ciphertext

		----- Calculations -----
		m = ct[0] + ( ct[1] * sk )
		pt = m * (t / q)

		----- Output -----
		pt	- Plaintext integer of ciphertext

		"""

		counter = self.counters['dec']

		# scaled_pt = ct[1]*sk + ct[0]
		scaled_pt = self.polyadd( self.polymult( ct[1], self.sk ), ct[0] )
		self.memory_access_polynomial(self.sk)
		self.memory_access_polynomial(ct[0])
		self.memory_access_polynomial(ct[1])
		counter['mul'] += len(self.sk)**2
		counter['add'] += len(self.sk)**2 + len(self.sk)
		counter['mod'] += len(self.sk)*2

		scaled_pt = scaled_pt * (self.t / self.q)
		scaled_pt.round()
		scaled_pt = scaled_pt % self.t
		decrypted_pt = scaled_pt
		counter['mul'] += len(scaled_pt)
		counter['mod'] += len(self.sk)

		return decrypted_pt

	def ctadd(self, x, y):
		"""
		Add two ciphertexts and return a ciphertext which
		should decrypt as the addition of plaintext inputs

		----- Arguments -----
		x	- Ciphertext for addition
		y	- Ciphertext for addition

		----- Calculation -----
		ct = x + y

		----- Output -----
		ct	- Ciphertext equivalent to x+y

		"""
		counter = self.counters['add']

		ct0 = self.polyadd(x[0],y[0])
		ct1 = self.polyadd(x[1],y[1])
		self.memory_access_polynomial(x[0])
		self.memory_access_polynomial(x[1])
		self.memory_access_polynomial(y[0])
		self.memory_access_polynomial(y[1])
		counter['add'] += len(x[1])*2
		counter['mod'] += len(x[1])*2

		ct = (ct0,ct1)

		return ct

	def mod(self,poly):
		"""	
		calculate the modulus of poly by q
		with answer given back in range (-q/2,q/2]
		"""
		copy = poly.poly.copy()
		for ind,i in enumerate(copy):
			i = i % self.q
			if ( i > (self.q/2) ):
				copy[ind] = i - self.q

		return Poly(copy)

	def polyadd(self, x, y):
		"""
		add two polynomials together and keep them 
		within the polynomial ring
		"""
		z = x + y
		quo,rem = (z // self.fn)
		z = rem

		z = z % self.q
		return z

	def polymult(self, x, y):
		"""
		multiply two polynomials together and keep them 
		within the polynomial ring
		"""

		z = x * y
		quo, rem = (z // self.fn)
		z = rem

		z = z % self.q
		return z

	def gen_normal_poly(self,c=0,std=None):
		"""
		generate a random polynomial of degree n-1
		with coefficients selected from normal distribution
		with center at 0 and std of 2. Each term is rounded
		down to nearest integer
		"""
		if ( std == None ):
			std = self.std
		a = []
		for i in range(self.n):
			a.append( int(np.random.normal(c,std)) )
		a = Poly(a)
		return a

	def gen_binary_poly(self):
		"""
		generate a random polynomial of degree n-1
		with coefficients ranging from [0,1]
		"""
		a = []
		for i in range(self.n):
			a.append( np.random.randint(0,2) )
		a = Poly(a)
		return a

	def gen_uniform_poly(self,q=None):
		"""
		generate a random polynomial of degree n-1
		with coefficients ranging from [0,q)
		"""
		if (q == None):
			q = self.q
		a = []
		for i in range(self.n):
			a.append( random.randint(0,q) )
		a = Poly(a)

		return a

	def print_counter_info(self):
		"""
		this function will print out the operation
		costs for each function (enc,dec,ctadd,ctmult...)
		"""

		print('Encryption OpCount')
		print( self.counters['enc'] )
		print('\nDecryption OpCount')
		print( self.counters['dec'] )
		print('\nAdd OpCount')
		print( self.counters['add'] )
		print('\nMemory Accesses')
		tab_width = ' '*2
		print(f'{tab_width}Cache Accesses: {self.cache_accesses}')
		print(f'{tab_width}Main Memory Accesses: {int(self.main_memory_accesses)}')
	
	def memory_access_polynomial(self, poly):
		poly_size = len(poly)
		int_byte_size = (self.bitwidth // 8) or 1

		ints_per_block = self.block_size // int_byte_size 
		if (self.block_size % int_byte_size != 0):
			ints_per_block += 1

		blocks_per_poly = poly_size // ints_per_block
		if (poly_size % ints_per_block != 0):
			blocks_per_poly += 1
		
		self.cache_accesses += poly_size
		self.main_memory_accesses += (poly_size * self.cache_miss_rate)


if __name__ == '__main__':
	main()
	pass