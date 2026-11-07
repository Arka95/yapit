"""Compatibility entry point for the enrollment module's native UI."""

from yapit.enrollment.desktop import EnrollmentApp, main

__all__ = ["EnrollmentApp", "main"]

if __name__ == "__main__":
    main()
