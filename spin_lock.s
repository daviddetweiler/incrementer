.global spin_lock
.global spin_unlock

spin_lock:
	jmp spin_lock_test

spin_lock_wait:
	pause

spin_lock_test:
	mov		$1,	%al
	xchg	%al, (%rdi)
	test	%al, %al
	je		spin_lock_wait

	ret

spin_unlock:
	movb 	$0, (%rdi)
	ret
