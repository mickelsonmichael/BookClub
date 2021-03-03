import { render, screen } from '@testing-library/react';
import App from './App';

test('mentions the name of the club', () => {
  render(<App />);
  const linkElement = screen.getByText(/dev book club/i);
  expect(linkElement).toBeInTheDocument();
});
