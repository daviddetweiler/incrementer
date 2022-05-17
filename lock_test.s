.global run_lock_test

.text
run_lock_test:
	mov		%rdi, %r8
	xor		%r9, %r9

begin:
	cmp		%r8, %r9
	je		done

	// Address index
	mov		%r9, %r10
	shl		$3, %r10
	add		%rdx, %r10

	// copy index
	mov		%r10, %rdi

	// Address cacheline
	mov		(%r10), %r10
	shl		$5, %r10
	add		%rsi, %r10

	// Address lock
	mov		(%rdi), %rdi
	shl		$5, %rdi
	add		%rcx, %rdi

	call	spin_lock
	incq	(%r10)
	call	spin_unlock
	
	inc		%r9
	jmp		begin

done:	
	ret
