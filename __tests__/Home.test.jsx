import React from 'react';
import { render, screen } from '@testing-library/react';
import Home from '../app/page';
import '@testing-library/jest-dom';

test('renders learn react link', () => {
  render(<Home />);
  const linkElement = screen.getByText(/Use AI instead/i);
  expect(linkElement).toBeInTheDocument();
});