import { Badge } from '@radix-ui/themes';

interface DealTypeBadgeProps {
  dealType?: 'rental_income' | 'value_add';
}

export function DealTypeBadge({ dealType }: DealTypeBadgeProps) {
  if (!dealType) return null;
  
  const config = {
    rental_income: {
      label: 'Rental Income',
      color: 'blue' as const,
    },
    value_add: {
      label: 'Value-Add',
      color: 'orange' as const,
    },
  };
  
  const { label, color } = config[dealType];
  
  return (
    <Badge size="2" variant="soft" color={color}>
      {label}
    </Badge>
  );
}
